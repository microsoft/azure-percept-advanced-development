// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <iostream>
#include <vector>

// Third-party includes
#include <opencv2/core/core.hpp>

// Local includes
#include "human_pose.hpp"

namespace pose {

HumanPose::HumanPose(const std::vector<cv::Point2f> &keypoints, const float &score)
    : keypoints(keypoints), score(score)
{
}

std::ostream& operator<<(std::ostream &os, const HumanPose &pose)
{
    os << pose.to_string();
    return os;
}

std::string HumanPose::to_string() const
{
    std::string s = "\"Keypoints\": [";
    for (const auto &point : this->keypoints)
    {
        s.append("{").append("\"x\": ").append(std::to_string(point.x)).append(", ").append("\"y\": ").append(std::to_string(point.y)).append("}, ");
    }
    // If there was at least one keypoint, remove the trailing comma and space
    if (this->keypoints.size() > 0)
    {
        s = s.substr(0, s.length() - 2);
    }
    s.append("]");

    // Append the confidence as well
    s.append(", ").append("\"Confidence\": ").append(std::to_string(this->score));

    return s;
}

} // namespace pose
