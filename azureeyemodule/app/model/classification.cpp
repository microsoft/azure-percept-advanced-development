// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <thread>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "classification.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/classification_kernels.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"


namespace model {

/** A classification network takes a single input and outputs a single output (which we will parse into labels and confidences) */
G_API_NET(ClassificationNetwork, <cv::GMat(cv::GMat)>, "classification-network");

ClassificationModel::ClassificationModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, labelfpath(labelfpath), class_labels({})
{
}

void ClassificationModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        this->wait_for_device();

        // Read in the labels for classification
        label::load_label_file(this->class_labels, this->labelfpath);
        this->log_parameters();

        // Build the camera pipeline with G-API
        *pipeline = this->compile_cv_graph();
        util::log_info("starting the pipeline...");
        pipeline->start();

        // Pull data through the pipeline
        bool ran_out_naturally = this->pull_data(*pipeline);

        if (!ran_out_naturally)
        {
            break;
        }
    }
}

cv::GStreamingCompiled ClassificationModel::compile_cv_graph() const
{
    // Declare an empty GMat - the beginning of the pipeline
    cv::GMat in;
    cv::GMat preproc = cv::gapi::mx::preproc(in, this->resolution);
    cv::GArray<uint8_t> h264;
    cv::GOpaque<int64_t> h264_seqno;
    cv::GOpaque<int64_t> h264_ts;
    std::tie(h264, h264_seqno, h264_ts) = cv::gapi::streaming::encH264ts(preproc);

    // We have BGR output and H264 output in the same graph.
    // In this case, BGR always must be desynchronized from the main path
    // to avoid internal queue overflow (FW reports this data to us via
    // separate channels)
    // copy() is required only to maintain the graph contracts
    // (there must be an operation following desync()). No real copy happens
    cv::GMat img = cv::gapi::copy(cv::gapi::streaming::desync(preproc));

    // This branch has inference and is desynchronized to keep
    // a constant framerate for the encoded stream (above)
    cv::GMat bgr = cv::gapi::streaming::desync(preproc);

    cv::GMat nn = cv::gapi::infer<ClassificationNetwork>(bgr);

    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(nn);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    cv::GArray<int> ids;
    cv::GArray<float> cfs;
    std::tie(ids, cfs) = cv::gapi::streaming::parseClass(nn, sz);

    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),
                                    cv::GOut(h264, h264_seqno, h264_ts,      // main path: H264 (~constant framerate)
                                    img,                                     // desynchronized path: BGR
                                    nn_seqno, nn_ts, ids, cfs));

    auto networks = cv::gapi::networks(cv::gapi::mx::Params<ClassificationNetwork>{this->modelfiles.at(0)});

    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseClass>());

    // Compile the graph in streamnig mode, set all the parameters
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline, and start processing
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool ClassificationModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    cv::optional<cv::Mat> out_bgr;

    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    cv::optional<cv::Mat> out_nn;
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<int>> out_labels;
    cv::optional<std::vector<float>> out_confidences;

    std::vector<int> last_labels;
    std::vector<float> last_confidences;
    cv::Mat last_bgr;

    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_nn_seqno, out_nn_ts, out_labels, out_confidences)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_nn, out_nn_ts, out_nn_seqno, out_labels, out_confidences, last_labels, last_confidences);
        this->handle_bgr_output(out_bgr, last_bgr, last_labels, last_confidences);

        if (restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

void ClassificationModel::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<int> &last_labels, const std::vector<float> &last_confidences)
{
    // BGR output: visualize and optionally display
    if (!out_bgr.has_value())
    {
        return;
    }

    last_bgr = *out_bgr;

    cv::Mat original_bgr;
    last_bgr.copyTo(original_bgr);

    rtsp::update_data_raw(last_bgr);
    preview(last_bgr, last_labels, last_confidences);

    if (this->status_msg.empty())
    {
        rtsp::update_data_result(last_bgr);
    }
    else
    {
        cv::Mat bgr_with_status;
        last_bgr.copyTo(bgr_with_status);

        util::put_text(bgr_with_status, this->status_msg);
        rtsp::update_data_result(bgr_with_status);
    }

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(original_bgr, last_confidences);
}

void ClassificationModel::preview(const cv::Mat &rgb, const std::vector<int> &labels, const std::vector<float> &confidences) const
{
    for (std::size_t i = 0; i < labels.size(); i++)
    {
        // color of a label
        int index = labels[i] % label::colors().size();

        cv::putText(rgb,
            util::get_label(labels[i], this->class_labels) + ": " + util::to_string_with_precision(confidences[i], 2),
            cv::Point(0, i * 20) + cv::Point(3, 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(label::colors().at(index)),
            2);
    }
}

void ClassificationModel::handle_inference_output(const cv::optional<cv::Mat> &out_nn, const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                                  const cv::optional<std::vector<int>> &out_labels,
                                                  const cv::optional<std::vector<float>> &out_confidences, std::vector<int> &last_labels,
                                                  std::vector<float> &last_confidences)
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // The below objects are on the same desynchronized path
    // and are coming together
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_labels.has_value());
    CV_Assert(out_confidences.has_value());

    // Compose a single message for each detection, which follows this schema:
    //
    // {
    //      "label": string. Class label of the detected object,
    //      "confidence": float. Confidence of the detection,
    //      "timestamp": int. Timestamp of the detection.
    // }
    //
    // Push each of these detection messages into a vector.
    std::vector<std::string> messages;
    for (std::size_t i = 0; i < out_labels->size(); i++)
    {
        auto label = util::get_label(out_labels.value()[i], this->class_labels);
        auto confidence = std::to_string(out_confidences.value()[i]);
        auto timestamp = std::to_string(*out_nn_ts);

        std::string str = std::string("{");
        str.append("\"label\": \"").append(label).append("\", ")
           .append("\"confidence\": \"").append(confidence).append("\", ")
           .append("\"timestamp\": \"").append(timestamp).append("\"")
           .append("}");

        messages.push_back(str);
    }

    // Wrap the detection messages into a list. The send_message() function will wrap it back into curly braces for you.
    std::string str = std::string("[");
    for (size_t i = 0; i < messages.size(); i++)
    {
        if (i > 0)
        {
            str.append(", ");
        }
        str.append(messages[i]);
    }
    str.append("]");

    // Send the message over IoT
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, str);
    this->log_inference(str);

    // Update the cached labels and confidences now that we have new ones.
    last_labels = std::move(*out_labels);
    last_confidences = std::move(*out_confidences);
}

void ClassificationModel::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += ", firmware: " + this->mvcmd + ", parser: classification, label: " + this->labelfpath + ", classes: " + std::to_string((int)this->class_labels.size());
    util::log_info(msg);
}

} // namespace model
