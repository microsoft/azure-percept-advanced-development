// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

// Standard library includes
#include <vector>

// Third-party includes
#include <opencv2/core/core.hpp>

namespace pose {

/**
 * A struct to encapsulate and represent a Human Pose - the keypoints and score.
 */
struct HumanPose {
    HumanPose(const std::vector<cv::Point2f> &keypoints = std::vector<cv::Point2f>(), const float &score = 0);

    std::vector<cv::Point2f> keypoints;
    float score;
};

} // namespace pose
