// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <vector>

// Third-party includes
#include <opencv2/core/core.hpp>

// Local includes
#include "human_pose.hpp"

namespace pose {

HumanPose::HumanPose(const std::vector<cv::Point2f> &keypoints, const float &score)
    : keypoints(keypoints), score(score) {}

} // namespace pose
