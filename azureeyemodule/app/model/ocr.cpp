// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <string>
#include <thread>
#include <vector>
#include <fstream>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "ocr.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/ocr_kernels.hpp"
#include "../ocr/decoder.hpp"
#include "../ocr/ocrvis.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/labels.hpp"
#include "../util/helper.hpp"

namespace model {

using GMat2 = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(TextDetection, <GMat2(cv::GMat)>, "sample.custom.text_detect");
G_API_NET(TextRecognition, <cv::GMat(cv::GMat)>,"sample.custom.text_recogn");

OCRModel::OCRModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
        :AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, OCRDecoder(ocr::TextDecoder {0, "0123456789abcdefghijklmnopqrstuvwxyz#", '#'})
        {
        }

void OCRModel::run(cv::GStreamingCompiled* pipeline)
{
    while(true)
    {

        this->wait_for_device();

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

cv::GStreamingCompiled OCRModel::compile_cv_graph() const
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

    // TODO: remove copy

    // This branch has inference and is desynchronized to keep
    // a constant framerate for the encoded stream (above)
    cv::GMat bgr = cv::gapi::streaming::desync(preproc);

     // Text recognition input size
    cv::Size in_rec_sz{ 120, 32 };

    cv::GMat link, segm;
    std::tie(link, segm) = cv::gapi::infer<TextDetection>(bgr);
    cv::GOpaque<cv::Size> size = cv::gapi::streaming::size(bgr);
    cv::GArray<cv::RotatedRect> rrs = cv::gapi::streaming::PostProcess::on(link, segm, size, 0.8f, 0.8f);
    cv::GArray<cv::GMat> td_labels = cv::gapi::streaming::CropLabels::on(bgr, rrs, in_rec_sz);
    cv::GArray<cv::GMat> text = cv::gapi::infer2<TextRecognition>(bgr, td_labels);

    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(link);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(link);


    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,      // main path: H264 (~constant framerate)
                                  img,                                     // desynchronized path: BGR
                                  nn_seqno, nn_ts, rrs, text));

    auto textdetection_net = cv::gapi::mx::Params<TextDetection> {modelfiles.at(0)}.cfgOutputLayers({"model/link_logits_/add", "model/segm_logits/add"});
    auto textrecognition_net = cv::gapi::mx::Params<TextRecognition> {modelfiles.at(1)};
    auto networks = cv::gapi::networks(textdetection_net, textrecognition_net);

    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::OCVPostProcess>(), cv::gapi::kernels<cv::gapi::streaming::OCVCropLabels>());

    // Compile the graph in streamnig mode, set all the parameters
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline, and start processing
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool OCRModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    cv::optional<cv::Mat> out_bgr;

    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<cv::RotatedRect>> out_txtrcs;
    cv::optional<std::vector<cv::Mat>> out_text;

    cv::Mat last_bgr;
    std::vector<cv::RotatedRect> last_rcs;
    std::vector<std::string> last_text;

    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    while (pipeline.pull( cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_nn_seqno, out_nn_ts, out_txtrcs, out_text)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
		this->handle_bgr_output(out_bgr, last_bgr, last_rcs, last_text);
        this->handle_inference_output(out_nn_ts, out_nn_seqno, out_txtrcs, last_rcs, out_text, last_text);

        if (this->restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

void OCRModel::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<cv::RotatedRect> &last_rcs, const std::vector<std::string> &last_text)
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
    this->preview(last_bgr, last_rcs, last_text);

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
    this->save_retraining_data(original_bgr);
}

void OCRModel::preview(cv::Mat &bgr, const std::vector<cv::RotatedRect> &last_rcs, const std::vector<std::string> &last_text) const
{

    /*Drawing text boxes and writing the text in BGR frame*/
    const auto num_labels = last_rcs.size();

    for( std::size_t l=0; l<num_labels; l++)
    {
        //Draw bounding box for this rotated rectangle
        const auto &rc = last_rcs[l];
        ocr::vis::drawRotatedRect(bgr, rc);

        // Draw text, if decoded
        ocr::vis::drawText(bgr, rc, last_text[l]);
    }
}

void OCRModel::handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                        const cv::optional<std::vector<cv::RotatedRect>> &out_txtrcs, std::vector<cv::RotatedRect> &last_rcs,
                                        cv::optional<std::vector<cv::Mat>> &out_text, std::vector<std::string> &last_text)
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // The below objects are on the same desynchronized path
    // and are coming together
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());

    // Handling text detection outputs
    CV_Assert(out_text.has_value());
    CV_Assert(out_txtrcs->size() == out_text->size());

    // Hold output in temp vars before pruning
    auto temp_text = *out_text;
    auto temp_rcs = *out_txtrcs;

    // To hold all the results from current text and rectangles
    std::vector<std::string> curr_textresults;
    std::vector<cv::RotatedRect> curr_rcsresults;

    // Collect all texts and send to IoT Hub
    std::string msg = "{\"Texts\": [";

    const auto num_labels = temp_rcs.size();
    for (std::size_t label_idx = 0; label_idx < num_labels; label_idx++)
    {
        // Decode the recognized text in the rectangle
        auto decoded = this->OCRDecoder.decode(temp_text[label_idx]);
        this->log_inference("Text: \"" + decoded.text + "\"");
        if (decoded.conf > 0.2)
        {
            curr_textresults.push_back(decoded.text);
            curr_rcsresults.push_back(temp_rcs[label_idx]);

            msg.append("\"" + decoded.text + "\", ");
        }
        else
        {
            msg.append("\"<COULD NOT DECODE>\", ");
        }
    }

    // If there was at least one label, we need to remove the trailing space and comma
    // because JSON is stupid and can't handle trailing commas. :/
    if (num_labels > 0)
    {
        msg = msg.substr(0, msg.length() - 2);
    }

    msg.append("]}");

    // Send all result into last_text and then dump all curr text results
    if(curr_textresults.size() > 0)
    {
        last_text = std::move(curr_textresults);
        last_rcs = std::move(curr_rcsresults);
    }
    else
    {
        last_text = {};
        last_rcs = {};
    }

    // Send resulting message over IoT
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, msg);
}

void OCRModel::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += ", firmware: " + this->mvcmd + ", parser: OpenPose";
    util::log_info(msg);
}

} // namespace model
