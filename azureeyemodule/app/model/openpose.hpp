// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include "azureeyemodel.hpp"
#include "../openpose/human_pose.hpp"


namespace model {

class OpenPoseModel : public AzureEyeModel
{
public:
    OpenPoseModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** Compile the pipeline graph for OpenPose. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    bool pull_data(cv::GStreamingCompiled &pipeline);

    /** Compose the BGR frame with the skeletons */
    void handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<pose::HumanPose> &poses);

    /** Compose the RGB based on the poses. */
    void preview(cv::Mat &bgr, const std::vector<pose::HumanPose> &poses) const;

    /** Handle the pose estimation model's output */
    void handle_inference_output(const cv::optional<cv::Mat> &out_nn, const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                                const cv::optional<std::vector<pose::HumanPose>> &out_poses, std::vector<pose::HumanPose> &last_poses);

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
