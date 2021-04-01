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
#include "binaryunet.hpp"
#include "../device/device.hpp"
#include "../util/helper.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/binaryunet_kernels.hpp"
#include "../streaming/rtsp.hpp"

namespace model {

/** A classification network takes a single input and outputs a single output (which we will parse into labels and confidences) */
G_API_NET(UNetNetwork, <cv::GMat(cv::GMat)>, "unet-network");

BinaryUnetModel::BinaryUnetModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel( modelfpaths, mvcmd, videofile, resolution )
{
}

void BinaryUnetModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        this->wait_for_device();
        this->log_parameters();

        // Build the camera pipeline with G-API
        *pipeline = this->compile_cv_graph();
        util::log_info("starting segmentation pipeline...");
        pipeline->start();

        // Pull data through the pipeline
        bool ran_out_naturally = this->pull_data(*pipeline);
        if (!ran_out_naturally)
        {
            break;
        }
    }
}

cv::GStreamingCompiled BinaryUnetModel::compile_cv_graph() const
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

    cv::GMat segmentation = cv::gapi::infer<UNetNetwork>(bgr);

    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    cv::GMat mask = cv::gapi::streaming::PostProcBinaryUnet::on(segmentation);

    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),
                                    cv::GOut(h264, h264_seqno, h264_ts,      // main path: H264 (~constant framerate)
                                    img,                                     // desynchronized path: BGR
                                    mask));

    auto networks = cv::gapi::networks(cv::gapi::mx::Params<UNetNetwork>{this->modelfiles.at(0)});

    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVPostProcBinaryUnet>());

    // Compile the graph in streamnig mode, set all the parameters
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the AzureEye's Camera as the input to the pipeline, and start processing
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    util::log_info("Succesfully compiled segmentation pipeline");
    return pipeline;
}

bool BinaryUnetModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    cv::optional<cv::Mat> out_bgr;

    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    cv::optional<cv::Mat> out_mask;

    cv::Mat last_bgr;
    cv::Mat last_mask;

    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    util::log_info("Pull_data: prep to pull");
    // Pull the data from the pipeline while it is running
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_mask)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_mask, last_mask, 0.5);
        this->handle_bgr_output(out_bgr, last_bgr, last_mask);

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

void BinaryUnetModel::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr,  const cv::Mat &last_mask)
{
    // BGR output: visualize and optionally display
    if (!out_bgr.has_value())
    {
        return;
    }

    last_bgr = *out_bgr;
    rtsp::update_data_raw(last_bgr);

    // draw on the bgr
    if(!last_mask.empty()) {
        preview(last_bgr, last_mask);
    }
    rtsp::update_data_result(last_bgr);
}

void BinaryUnetModel::preview(cv::Mat &frame, const cv::Mat& last_mask) const
{
      // create a BGR mask
      cv::Mat g_prob = last_mask > 0;
      cv::Mat g;
      g_prob.convertTo(g, CV_8U);

      cv::Mat r = (1 - g);
      cv::Mat b(r.size(), CV_8UC1, cv::Scalar(0));

      std::vector<cv::Mat> channels{ b, g, r };
      cv::Mat show_mask;
      cv::merge(channels, show_mask);
      show_mask *= 255;

      cv::resize(show_mask, show_mask, frame.size());

      // These are blending coefficients for adding the mask
      // to the image: alpha * image + beta * mask
      float alpha = 0.8;
      float beta = 1 - alpha;

      cv::Mat dst;
      cv::addWeighted(frame, alpha, show_mask, beta, 0.0, dst);
      dst.copyTo(frame);
}

void BinaryUnetModel::handle_inference_output(const cv::optional<cv::Mat> &out_mask, cv::Mat &last_mask, float threshold)
{
    if (!out_mask.has_value())
    {
        return;
    }

    // Create a mask - everywhere that the network has confidence greater than threshold
    cv::Mat mask_vals(*out_mask > threshold);

    // Compute the fraction of the image that is occupied by detections
    float relativeOccupied = static_cast<float>(cv::countNonZero(mask_vals)) / (mask_vals.rows * mask_vals.cols);

    // Create JSON message of the following schema
    //
    // [
    //   {
    //     "occupied": <float> fraction of the image covered by detections
    //   }
    // ]
    //
    // The outer list is not strictly necessary, but because most (all?) of the other
    // networks have multiple detections per output, they all use lists, so this one
    // uses a list just to conform.
    std::string str = std::string("[");
    str.append("{")
        .append("\"occupied\": \"").append(std::to_string(relativeOccupied)).append("\"")
        .append("}");
    str.append("]");
    last_mask = std::move(mask_vals);
    this->log_inference(str);

    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, str);
}

void BinaryUnetModel::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += "firmware: " + this->mvcmd;
    util::log_info(msg);
}

} // namespace model
