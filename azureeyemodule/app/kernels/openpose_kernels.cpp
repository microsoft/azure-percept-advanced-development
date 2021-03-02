// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <vector>

// Third party includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

// Local includes
#include "openpose_kernels.hpp"
#include "../openpose/peak.hpp"

namespace cv {
namespace gapi {
namespace streaming {

GPoses parse_open_pose(const GMat &pafs, const GMat &keys, const GOpaque<Size> &in_sz)
{
    return GParseOpenPose::on(pafs, keys, in_sz);
}

void extract_poses(const std::vector<cv::Mat> &heat_maps, const std::vector<cv::Mat> &pafs, std::vector<pose::HumanPose> &poses,
                   const float min_peaks_distance, const size_t keypoints_number, const float mid_points_score_threshold,
                   const float found_mid_points_ratio_threshold, const int min_joints_number, const float min_subset_score)
{
    // Do a CV CPU-parallelized loop to compute all the peaks in all the heat maps. Peaks are possible key points.
    std::vector<std::vector<pose::peak::Peak>> peaks_from_heat_map(heat_maps.size());
    FindPeaksBody find_peaks_body(heat_maps, min_peaks_distance, peaks_from_heat_map);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heat_maps.size())), find_peaks_body);

    // All peaks in heat map 0 have id = -1
    // All peaks in heat map 1 have id = n peaks in heat map 0
    // All peaks in heat map 2 have id = (n peaks in heat map 0) + (n peaks in heat map 1)
    // Etc.
    int peaks_before = 0;
    for (size_t heat_map_id = 1; heat_map_id < heat_maps.size(); heat_map_id++)
    {
        peaks_before += static_cast<int>(peaks_from_heat_map[heat_map_id - 1].size());
        for (auto &peak : peaks_from_heat_map[heat_map_id]) {
            peak.id += peaks_before;
        }
    }

    // Use the peaks (possible keypoints) to fill in the HumanPose vector
    poses.clear();
    pose::peak::group_peaks_to_poses(peaks_from_heat_map, pafs, keypoints_number, mid_points_score_threshold, found_mid_points_ratio_threshold, min_joints_number, min_subset_score, poses);
}

void correct_coordinates(std::vector<pose::HumanPose> &poses, const cv::Size &feature_map_size, const cv::Size &image_size, int stride, int upsample_ratio)
{
    CV_Assert(stride % upsample_ratio == 0);

    cv::Vec4i pad(cv::Vec4i::all(0));

    cv::Size full_feature_map_size = feature_map_size * stride / upsample_ratio;

    float scaleX = image_size.width / static_cast<float>(full_feature_map_size.width - pad(1) - pad(3));
    float scaleY = image_size.height / static_cast<float>(full_feature_map_size.height - pad(0) - pad(2));
    for (auto &pose : poses)
    {
        for (auto &keypoint : pose.keypoints)
        {
            if (keypoint != cv::Point2f(-1, -1))
            {
                keypoint.x *= stride / upsample_ratio;
                keypoint.x -= pad(1);
                keypoint.x *= scaleX;

                keypoint.y *= stride / upsample_ratio;
                keypoint.y -= pad(0);
                keypoint.y *= scaleY;
            }
        }
    }
}

} // namespace streaming
} // namespace gapi
} // namespace cv
