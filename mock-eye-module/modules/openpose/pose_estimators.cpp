// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <string>
#include <vector>

// Third party library includes
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

// Our includes
#include "../../kernels/openpose_kernels.hpp"
#include "../../kernels/utils.hpp"
#include "../device.hpp"
#include "human_pose.hpp"
#include "pose_estimators.hpp"
#include "util.hpp"

namespace pose {

/** Declare an OpenPose network type. Takes one matrix and outputs two matrices. */
using openpose_output = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(OpenPose, <openpose_output(cv::GMat)>, "com.microsoft.azure.open-pose");

static cv::GStreamingCompiled build_inference_graph(const std::string &video_fpath, const std::string &modelfile, const std::string &weightsfile, const device::Device &device)
{
    // Build the pipeline.
    cv::GMat in;

    // Apply the network to the frame to produce a single output image
    cv::GMat nn_pafs;
    cv::GMat nn_keys;
    std::tie(nn_pafs, nn_keys) = cv::gapi::infer<OpenPose>(in);
    cv::GOpaque<cv::Size> sz = cv::gapi::custom::size(in);
    cv::GMat bgr = cv::gapi::copy(in);

    // Parse the output of the network pipeline based on the type of network.
    cv::GArray<HumanPose> skeletons = cv::gapi::custom::parse_open_pose(nn_pafs, nn_keys, sz);

    // Set up the actual kernels (the implementations of the parser ops)
    auto kernels = cv::gapi::combine(cv::gapi::kernels<cv::gapi::custom::GOCVParseOpenPose>(),
                                     cv::gapi::kernels<cv::gapi::custom::GOCVSize>(),
                                     cv::gapi::kernels<cv::gapi::custom::GOCVSizeR>());

    // Instantiate the network from the files (this is also where we instantiate our chosen backend)
    auto net = cv::gapi::ie::Params<OpenPose>{ modelfile, weightsfile, device_to_string(device) }.cfgOutputLayers({"Mconv7_stage2_L1", "Mconv7_stage2_L2"});

    // Set up the inputs and outputs of the graph
    auto comp = cv::GComputation(cv::GIn(in), cv::GOut(bgr, nn_pafs, nn_keys, skeletons));
    auto compiled_args = cv::compile_args(cv::gapi::networks(net), kernels);
    auto cs = comp.compileStreaming(std::move(compiled_args));

    // Set the source to the graph
    if (video_fpath.empty())
    {
        // Specify the web cam as the input to the pipeline.
        cs.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(-1));
    }
    else
    {
        // Specify the user-supplied video file as the input to the pipeline.
        auto gstcmd = "filesrc location=" + video_fpath + " ! decodebin ! videoconvert ! appsink";
        cs.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(gstcmd));
    }

    return cs;
}

/**
 * Overlay stuff on top of the BGR image to show the results.
 */
static void preview(cv::Mat &bgr, const std::vector<HumanPose> &poses)
{
    CV_Assert(bgr.type() == CV_8UC3);

    static const cv::Scalar colors[] = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
    };
    static const std::pair<int, int> limb_keypoints_ids[] = {
        {1, 2},  {1, 5},   {2, 3},
        {3, 4},  {5, 6},   {6, 7},
        {1, 8},  {8, 9},   {9, 10},
        {1, 11}, {11, 12}, {12, 13},
        {1, 0},  {0, 14},  {14, 16},
        {0, 15}, {15, 17}
    };

    const int stick_width = 4;
    const cv::Point2f absent_keypoint(-1.0f, -1.0f);

    // For each pose
    for (const auto &pose : poses)
    {
        CV_Assert(pose.keypoints.size() == array_size(colors));

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

/**
 * Pulls the data from the graph pipeline forever.
 *
 * @param cs The compiled streaming graph
 * @param opt_show If true, we visualize the display (requires a GUI)
 */
void pull_data_from_pipeline(cv::GStreamingCompiled &cs, bool opt_show)
{
    // Set up all the output nodes
    cv::Mat out_bgr;
    cv::Mat out_pafs;
    cv::Mat out_keys;
    std::vector<HumanPose> out_poses;

    // This while loop is where we pull the data from the camera and output to the RTSP stream
    // i.e., this is where we actually run the pipeline
    while (cs.pull(cv::gout(out_bgr, out_pafs, out_keys, out_poses)))
    {
        // BGR output: visualize and optionally display
        preview(out_bgr, out_poses);

        // Preview if humans are watching us run this on a GUI
        if (opt_show)
        {
            cv::imshow("preview", out_bgr);
            cv::waitKey(1);
        }
    }
}

void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show)
{
    auto cs = build_inference_graph(video_fpath, modelfpath, weightsfpath, device);
    cs.start();
    pull_data_from_pipeline(cs, show);
}

} // namespace pose
