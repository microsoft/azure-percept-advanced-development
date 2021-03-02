/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * Based on some code from Intel:
 *
 * Copyright (C) 2018-2019 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * A peak is a peak value in a heat map. A peak is a possible keypoint.
 */

#pragma once

// Standard library includes
#include <vector>

// Third-party library includes
#include <opencv2/core/core.hpp>

// Local includes
#include "human_pose.hpp"


namespace pose {
namespace peak {

/**
 * A peak in OpenPose.
 *
 * A peak is a peak in a heat map, and should correspond to a single keypoint, though
 * this keypoint may be a false positive.
 */
struct Peak {
    Peak(const int id = -1, const cv::Point2f& pos = cv::Point2f(), const float score = 0.0f);

    /** Ids are unique across what? */
    int id;

    /** The pixel location in the heatmap of this peak (keypoint) */
    cv::Point2f pos;

    /** I assume this is the magnitude of the heat map at this peak. */
    float score;
};

/**
 * A HumanPose before it has been converted into a true HumanPose object.
 *
 * This object is simply a collection of peaks across the heat maps.
 */
struct HumanPoseByPeaksIndices {
    explicit HumanPoseByPeaksIndices(const int keypoints_number);

    /** The particular peaks that make up this 'human'. */
    std::vector<int> peaks_indices;
    /** TODO: Don't know: Maybe the number of joints in this 'human' */
    int njoints;
    /** TODO: Don't know */
    float score;
};

/**
 * A struct to house a pair of key points that are connected by a weighted edge.
 */
struct TwoJointsConnection {
    TwoJointsConnection(const int first_joint_idx, const int second_joint_idx, const float score);

    /** The ID of the first joint in this connection. */
    int first_joint_idx;
    /** The ID of the second joint in this connection. */
    int second_joint_idx;
    /** The score for this edge. I'm guessing this is the approximated line integral across the PAFs from first to second joint. */
    float score;
};

/**
 * Finds all the peaks in the given `heat_maps[heat_map_id]`.
 * This is the worker kernel for a parallelized implementation of finding all the peaks in all the heat maps,
 * hence why we just pick out a single heat map.
 *
 * @param heat_maps: The heat maps
 * @param min_peaks_distance: If we find two peaks that are not at least this far apart, we combine them into a single one.
 * @param all_peaks: TODO: I assume we fill this in in this function?
 * @param heat_map_id: TODO: I assume we are only finding the peaks from `heat_maps[heat_map_id]`?
 */
void find_peaks(const std::vector<cv::Mat> &heat_maps, const float min_peaks_distance, std::vector<std::vector<Peak>> &all_peaks, int heat_map_id);

/**
 * Produces HumanPose objects - one per person in the image, based on the peaks found in the heat maps.
 *
 * Peaks in the heat maps correspond to possible keypoints. But they need to be compiled together into a list
 * of graphs, each graph represents a single skeleton. This is the function that does this.
 *
 * @param all_peaks: All the peaks - a list of all peaks in a particular heat map, for each heat map.
 * @param pafs: The part affinity fields from the neural network. We use this to weight peak combinations.
 * @param key_points_number: The number of key points on each skeleton.
 * @param mid_points_score_threshold: TODO: No idea
 * @param found_mid_points_ratio_threshold: TODO: No idea
 * @param min_joints_number: For each subset of joints (possible skeleton) that we come up with, we exclude it if
 *                           its number of joints is less than this.
 * @param min_subset_score: For each subset of joints (possible skeleton) that we come up with, we exclude it if
 *                          its score over njoints is less than this value.
 * @param poses: The vector of poses to fill.
 */
void group_peaks_to_poses(const std::vector<std::vector<Peak>> &all_peaks, const std::vector<cv::Mat> &pafs, const size_t keypoints_number,
                          const float mid_points_score_threshold, const float found_mid_points_ratio_threshold, const int min_joints_number,
                          const float min_subset_score, std::vector<HumanPose> &poses);

} // namespace peak
} // namespace pose
