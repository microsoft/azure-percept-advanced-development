// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <iostream>
#include <vector>

// Third-party includes
#include <opencv2/core/core.hpp>

namespace pose {

/**
 * A struct to encapsulate and represent a Human Pose - the keypoints and score.
 */
struct HumanPose {
    HumanPose(const std::vector<cv::Point2f> &keypoints = std::vector<cv::Point2f>(), const float &score = 0);

    /** Return a string representation of this struct. */
    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream &os, const HumanPose &pose);

    std::vector<cv::Point2f> keypoints;
    float score;
};

} // namespace pose
