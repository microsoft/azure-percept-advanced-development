// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <assert.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "openpose.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/openpose_kernels.hpp"
#include "../openpose/human_pose.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace model {

/** Colors we use for OpenPose viewing */
static const cv::Scalar colors[] = {
    cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
    cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
    cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
    cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
};

/** Keypoint pairs that make up 'limbs' */
static const std::pair<int, int> limb_keypoints_ids[] = {
    {1, 2},  {1, 5},   {2, 3},
    {3, 4},  {5, 6},   {6, 7},
    {1, 8},  {8, 9},   {9, 10},
    {1, 11}, {11, 12}, {12, 13},
    {1, 0},  {0, 14},  {14, 16},
    {0, 15}, {15, 17}
};

/** Width of the 'limbs' in viewing stuff */
const int stick_width = 4;

/** Declare an OpenPose network type. Takes one matrix and outputs two matrices. */
using openpose_output = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(OpenPoseNetwork, <openpose_output(cv::GMat)>, "com.intel.azure.open-pose");

OpenPoseModel::OpenPoseModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }
{
}

void OpenPoseModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
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

cv::GStreamingCompiled OpenPoseModel::compile_cv_graph() const
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

    cv::GMat nn_pafs;
    cv::GMat nn_keys;
    std::tie(nn_pafs, nn_keys) = cv::gapi::infer<OpenPoseNetwork>(bgr);

    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn_keys);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(nn_keys);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    cv::GArray<pose::HumanPose> skeletons = cv::gapi::streaming::parse_open_pose(nn_pafs, nn_keys, sz);

    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,               // main path: H264 (~constant framerate)
                                  img,                                              // desynchronized path: BGR
                                  nn_seqno, nn_ts, nn_pafs, nn_keys, skeletons));   // Inference path

    auto networks = cv::gapi::networks(cv::gapi::mx::Params<OpenPoseNetwork>{ this->modelfiles.at(0) }.cfgOutputLayers({"Mconv7_stage2_L1", "Mconv7_stage2_L2"}));

    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseOpenPose>());

    // Compile the graph in streamnig mode, set all the parameters
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline, and start processing
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool OpenPoseModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    cv::optional<cv::Mat> out_bgr;

    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    cv::optional<cv::Mat> out_pafs;
    cv::optional<cv::Mat> out_keys;
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<pose::HumanPose>> out_poses;

    std::vector<int> last_labels;
    std::vector<float> last_confidences;
    std::vector<pose::HumanPose> last_poses;
    cv::Mat last_bgr;

    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_nn_seqno, out_nn_ts, out_pafs, out_keys, out_poses)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_keys, out_nn_ts, out_nn_seqno, out_poses, last_poses);
        this->handle_bgr_output(out_bgr, last_bgr, last_poses);

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

void OpenPoseModel::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<pose::HumanPose> &last_poses)
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
    this->preview(last_bgr, last_poses);

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

void OpenPoseModel::preview(cv::Mat &bgr, const std::vector<pose::HumanPose> &poses) const
{
    CV_Assert(bgr.type() == CV_8UC3);

    const cv::Point2f absent_keypoint(-1.0f, -1.0f);

    // For each pose
    for (const auto &pose : poses)
    {
        assert(pose.keypoints.size() == util::array_size(colors));

        // For each keypoint
        for (size_t keypoint_idx = 0; keypoint_idx < pose.keypoints.size(); keypoint_idx++)
        {
            // If the keypoint is not absent
            if (pose.keypoints[keypoint_idx] != absent_keypoint)
            {
                // Draw a circle on it
                cv::circle(bgr, pose.keypoints[keypoint_idx], 4, colors[keypoint_idx], -1);
            }
        }
    }

    cv::Mat pane = bgr.clone();
    // For each pose
    for (const auto &pose : poses)
    {
        // For each limb
        for (const auto &limb_id : limb_keypoints_ids)
        {
            // Don't draw the limb if either of its joint keypoints is missing
            std::pair<cv::Point2f, cv::Point2f> limb_keypoints(pose.keypoints[limb_id.first], pose.keypoints[limb_id.second]);
            if ((limb_keypoints.first == absent_keypoint) || (limb_keypoints.second == absent_keypoint))
            {
                continue;
            }

            float meanX = (limb_keypoints.first.x + limb_keypoints.second.x) / 2;
            float meanY = (limb_keypoints.first.y + limb_keypoints.second.y) / 2;
            cv::Point difference = limb_keypoints.first - limb_keypoints.second;
            double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
            int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stick_width), angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(pane, polygon, colors[limb_id.second]);
        }
    }
    cv::addWeighted(bgr, 0.4, pane, 0.6, 0, bgr);
}

void OpenPoseModel::handle_inference_output(const cv::optional<cv::Mat> &out_nn, const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                            const cv::optional<std::vector<pose::HumanPose>> &out_poses, std::vector<pose::HumanPose> &last_poses)
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // The below objects are on the same desynchronized path
    // and are coming together
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_poses.has_value());

    bool at_least_one_pose = false;
    std::string msg = "{\"Poses\": [";
    for (const auto &pose : out_poses.value())
    {
        msg.append("{").append(pose.to_string()).append("}, ");
        at_least_one_pose = true;
    }

    // If there is at least one pose, we need to remove the trailing space and comma
    if (at_least_one_pose)
    {
        msg = msg.substr(0, msg.length() - 2);
    }
    msg.append("]}");

    last_poses = std::move(*out_poses);

    this->log_inference(msg);
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, msg);
}

void OpenPoseModel::log_parameters() const
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
