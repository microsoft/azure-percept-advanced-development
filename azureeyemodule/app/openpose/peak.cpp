// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Based on some code from Intel under Apache-2.0:
// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Standard library includes
#include <algorithm>
#include <utility>
#include <vector>

// Third party includes
#include <opencv2/core/core.hpp>

// Local includes
#include "peak.hpp"
#include "../util/helper.hpp"

namespace pose {
namespace peak {

#define EXTRA_LIMB_IDX0 17
#define EXTRA_LIMB_IDX1 18
#define N_PAF_LIMB_PAIRS 19
#define N_HEATMAP_LIMB_PAIRS 19

Peak::Peak(const int id, const cv::Point2f &pos, const float score)
    : id(id), pos(pos), score(score)
{
    // Nothing to do
}

HumanPoseByPeaksIndices::HumanPoseByPeaksIndices(const int keypoints_number)
    : peaks_indices(std::vector<int>(keypoints_number, -1)), njoints(0), score(0.0f)
{
    // Nothing to todo
}

TwoJointsConnection::TwoJointsConnection(const int first_joint_idx, const int second_joint_idx, const float score)
    : first_joint_idx(first_joint_idx), second_joint_idx(second_joint_idx), score(score)
{
    // Nothing to do
}

void find_peaks(const std::vector<cv::Mat> &heat_maps, const float min_peaks_distance, std::vector<std::vector<Peak> > &all_peaks, int heat_map_id)
{
    const float threshold = 0.1f;

    // Collect all the peaks in each heat map.
    // A pixel is a peak if its value is greater than all its immediately adjacent pixels.

    // Collect all peaks across all heat maps into this one vector of 2D points (pixel locations)
    std::vector<cv::Point> peaks;

    // Index into the particular heat map this 'thread' will be filling in
    const cv::Mat &heat_map = heat_maps[heat_map_id];

    // Convert the heat map into a flat float buffer
    const float *heat_map_data = heat_map.ptr<float>();
    size_t heat_map_step = heat_map.step1();

    // For each row and column in the heat map
    for (int y = -1; y < heat_map.rows + 1; y++)
    {
        for (int x = -1; x < heat_map.cols + 1; x++)
        {
            // Determine if pixel at (x, y) is a peak value - a possible keypoint
            float val = 0;
            if ((x >= 0) && (y >= 0) && (x < heat_map.cols) && (y < heat_map.rows))
            {
                val = heat_map_data[y * heat_map_step + x];
                val = val >= threshold ? val : 0;
            }

            float left_val = 0;
            if ((y >= 0) && (x < (heat_map.cols - 1)) && (y < heat_map.rows))
            {
                left_val = heat_map_data[y * heat_map_step + x + 1];
                left_val = left_val >= threshold ? left_val : 0;
            }

            float right_val = 0;
            if ((x > 0) && (y >= 0) && (y < heat_map.rows))
            {
                right_val = heat_map_data[y * heat_map_step + x - 1];
                right_val = right_val >= threshold ? right_val : 0;
            }

            float top_val = 0;
            if ((x >= 0) && (x < heat_map.cols) && (y < (heat_map.rows - 1)))
            {
                top_val = heat_map_data[(y + 1) * heat_map_step + x];
                top_val = top_val >= threshold ? top_val : 0;
            }

            float bottom_val = 0;
            if ((x >= 0) && (y > 0) && (x < heat_map.cols))
            {
                bottom_val = heat_map_data[(y - 1) * heat_map_step + x];
                bottom_val = bottom_val >= threshold ? bottom_val : 0;
            }

            if ((val > left_val) && (val > right_val) && (val > top_val) && (val > bottom_val))
            {
                peaks.push_back(cv::Point(x, y));
            }
        }
    }

    // Sort all the peaks by their x coordinate.
    std::sort(peaks.begin(), peaks.end(), [](const cv::Point &a, const cv::Point &b) {
        return a.x < b.x;
    });

    // Determine which peaks are truly peaks
    // Peaks which are too close together are collapsed into a single one.
    std::vector<bool> is_actual_peak(peaks.size(), true);
    int peak_counter = 0;
    std::vector<Peak> &peaks_with_score_and_id = all_peaks[heat_map_id];
    for (size_t i = 0; i < peaks.size(); i++)
    {
        if (is_actual_peak[i])
        {
            for (size_t j = i + 1; j < peaks.size(); j++)
            {
                if (sqrt((peaks[i].x - peaks[j].x) * (peaks[i].x - peaks[j].x) + (peaks[i].y - peaks[j].y) * (peaks[i].y - peaks[j].y)) < min_peaks_distance)
                {
                    is_actual_peak[j] = false;
                }
            }
            peaks_with_score_and_id.push_back(Peak(peak_counter++, peaks[i], heat_map.at<float>(peaks[i])));
        }
    }
}

static inline void update_possible_poses_with_extra_connections(const std::vector<TwoJointsConnection> &connections, std::vector<HumanPoseByPeaksIndices> &possible_poses,
                                                                const int idx_jointA, const int idx_jointB)
{
    for (auto &connection : connections)
    {
        const int &indexA = connection.first_joint_idx;
        const int &indexB = connection.second_joint_idx;
        for (size_t j = 0; j < possible_poses.size(); j++)
        {
            if ((possible_poses[j].peaks_indices[idx_jointA] == indexA) && (possible_poses[j].peaks_indices[idx_jointB] == -1))
            {
                possible_poses[j].peaks_indices[idx_jointB] = indexB;
            }
            else if ((possible_poses[j].peaks_indices[idx_jointB] == indexB) && (possible_poses[j].peaks_indices[idx_jointA] == -1))
            {
                possible_poses[j].peaks_indices[idx_jointA] = indexA;
            }
        }
    }
}

static inline void update_possible_poses(const std::vector<TwoJointsConnection> &connections, std::vector<HumanPoseByPeaksIndices> &possible_poses,
                                         const int idx_jointA, const int idx_jointB, const std::vector<Peak> &candidates, const size_t keypoints_number)
{
    // For each connection/limb
    for (auto &connection : connections)
    {
        const int& indexA = connection.first_joint_idx;
        const int& indexB = connection.second_joint_idx;
        bool updated = false;
        for (size_t j = 0; j < possible_poses.size(); j++)
        {
            // For each possible pose
            if (possible_poses[j].peaks_indices[idx_jointA] == indexA)
            {
                // If that pose contains an A joint,
                // we update it to also include a B joint now (and update the total score for the pose)
                possible_poses[j].peaks_indices[idx_jointB] = indexB;
                possible_poses[j].njoints++;
                possible_poses[j].score += candidates[indexB].score + connection.score;
                updated = true;
            }
        }

        if (!updated)
        {
            // If we did not add either of these joints to any possible poses, create a new possible pose out of these
            // two joints
            HumanPoseByPeaksIndices newpose(keypoints_number);
            newpose.peaks_indices[idx_jointA] = indexA;
            newpose.peaks_indices[idx_jointB] = indexB;
            newpose.njoints = 2;
            newpose.score = candidates[indexA].score + candidates[indexB].score + connection.score;
            possible_poses.push_back(newpose);
        }
    }
}

static inline std::vector<HumanPoseByPeaksIndices> fill_possible_poses(const std::vector<TwoJointsConnection> &connections, const size_t keypoints_number,
                                                                       const int idx_jointA, const int idx_jointB, const std::vector<Peak> &candidates)
{
    auto possible_poses = std::vector<HumanPoseByPeaksIndices>(connections.size(), HumanPoseByPeaksIndices(keypoints_number));
    for (size_t i = 0; i < connections.size(); i++)
    {
        const int &indexA = connections[i].first_joint_idx;
        const int &indexB = connections[i].second_joint_idx;
        possible_poses[i].peaks_indices[idx_jointA] = indexA;
        possible_poses[i].peaks_indices[idx_jointB] = indexB;
        possible_poses[i].njoints = 2;
        possible_poses[i].score = candidates[indexA].score + candidates[indexB].score + connections[i].score;
    }

    return possible_poses;
}

static inline std::vector<TwoJointsConnection> take_most_promising_joint_connections(const std::vector<TwoJointsConnection> &temp_joint_connections,
                                                                                     const size_t njointsA, const size_t njointsB,
                                                                                     const std::vector<Peak> &candidates_for_jointA,
                                                                                     const std::vector<Peak> &candidates_for_jointB)
{
    std::vector<TwoJointsConnection> connections;

    // The maximum number of limbs we can create is equal to whichever heat map has the fewest peaks
    size_t num_limbs = std::min(njointsA, njointsB);

    size_t cnt = 0;
    std::vector<int> occurA(njointsA, 0);
    std::vector<int> occurB(njointsB, 0);
    for (auto &limb: temp_joint_connections)
    {
        if (cnt == num_limbs)
        {
            // We've taken as many temp limbs as we can, the rest are garbage.
            // Break from this loop.
            break;
        }

        // If we don't already have this connection in the adjacency matrix, create it and update the matrix
        const int &indexA = limb.first_joint_idx;
        const int &indexB = limb.second_joint_idx;
        const float &score = limb.score;
        if ((occurA[indexA] == 0) && (occurB[indexB] == 0))
        {
            connections.push_back(TwoJointsConnection(candidates_for_jointA[indexA].id, candidates_for_jointB[indexB].id, score));
            cnt++;
            occurA[indexA] = 1;
            occurB[indexB] = 1;
        }
    }

    return connections;
}

static inline void maybe_create_joint_connection(std::vector<TwoJointsConnection> &temp_joint_connections, const Peak &candidateA, const Peak &candidateB,
                                                 const std::pair<cv::Mat, cv::Mat> &score_mid, const std::vector<cv::Mat> &pafs,
                                                 const float mid_points_score_threshold, const float found_mid_points_ratio_threshold, const size_t idxA,
                                                 const size_t idxB)
{
    // Calculate the normalized vector pointing from A to B
    cv::Point2f pt = candidateA.pos * 0.5 + candidateB.pos * 0.5;
    cv::Point mid = cv::Point(cvRound(pt.x), cvRound(pt.y));
    cv::Point2f vec = candidateB.pos - candidateA.pos;
    double norm_vec = cv::norm(vec);
    if (norm_vec == 0)
    {
        return;
    }
    vec /= norm_vec;

    // Compute the score for this possible limb by computing an approximate (sampled) line integral along the vector
    float score = vec.x * score_mid.first.at<float>(mid) + vec.y * score_mid.second.at<float>(mid);
    int height_n  = pafs[0].rows / 2;
    float suc_ratio = 0.0f;
    float mid_score = 0.0f;
    const int mid_num = 10;
    const float score_threshold = -100.0f;

    if (score > score_threshold)
    {
        float p_sum = 0;
        int p_count = 0;
        cv::Size2f step((candidateB.pos.x - candidateA.pos.x)/(mid_num - 1), (candidateB.pos.y - candidateA.pos.y)/(mid_num - 1));
        for (int n = 0; n < mid_num; n++)
        {
            cv::Point midPoint(cvRound(candidateA.pos.x + n * step.width), cvRound(candidateA.pos.y + n * step.height));
            cv::Point2f pred(score_mid.first.at<float>(midPoint), score_mid.second.at<float>(midPoint));
            score = vec.x * pred.x + vec.y * pred.y;
            if (score > mid_points_score_threshold)
            {
                p_sum += score;
                p_count++;
            }
        }
        suc_ratio = static_cast<float>(p_count) / static_cast<float>(mid_num);
        float ratio = p_count > 0 ? p_sum / p_count : 0.0f;
        mid_score = ratio + static_cast<float>(std::min(height_n / norm_vec - 1, 0.0));
    }

    if ((mid_score > 0) && (suc_ratio > found_mid_points_ratio_threshold))
    {
        temp_joint_connections.push_back(TwoJointsConnection(idxA, idxB, mid_score));
    }
}

/**
 * For all possible combinations between joint type A and joint type B,
 * possibly create a temporary connection between them.
 *
 * Whether we do or not is controlled in some fashion by found_mid_points_ratio_threshold
 * and mid_points_score_threshold.
 */
static inline std::vector<TwoJointsConnection> create_temp_joint_connections(const size_t njointsA, const size_t njointsB,
                                                                             const std::vector<Peak> &candidates_for_jointA,
                                                                             const std::vector<Peak> &candidates_for_jointB,
                                                                             const std::pair<cv::Mat, cv::Mat> &score_mid,
                                                                             const std::vector<cv::Mat> &pafs, const float mid_points_score_threshold,
                                                                             const float found_mid_points_ratio_threshold)
{
    std::vector<TwoJointsConnection> temp_joint_connections;
    for (size_t i = 0; i < njointsA; i++)
    {
        for (size_t j = 0; j < njointsB; j++)
        {
            const Peak &candidateA = candidates_for_jointA[i];
            const Peak &candidateB = candidates_for_jointB[j];

            maybe_create_joint_connection(temp_joint_connections, candidateA, candidateB, score_mid, pafs, mid_points_score_threshold, found_mid_points_ratio_threshold, i, j);
        }
    }

    return temp_joint_connections;
}

static inline void create_poses_from_connections(const std::vector<TwoJointsConnection> &connections, const size_t limb_idx, std::vector<HumanPoseByPeaksIndices> &possible_poses,
                                                 const size_t keypoints_number, const int idx_jointA, const int idx_jointB, const std::vector<Peak> &candidates)
{
    // If there aren't any good connections, just quit
    if (connections.empty())
    {
        return;
    }

    if (limb_idx == 0)
    {
        // If this is the first limb, the possible_poses vector is empty,
        // so we fill it with each of the remaining connections: one connection (limb) per pose
        possible_poses = fill_possible_poses(connections, keypoints_number, idx_jointA, idx_jointB, candidates);
    }
    else if ((limb_idx == EXTRA_LIMB_IDX0) || (limb_idx == EXTRA_LIMB_IDX1))
    {
        // These are extra limbs
        update_possible_poses_with_extra_connections(connections, possible_poses, idx_jointA, idx_jointB);
    }
    else
    {
        // Update our possible poses to include these joints and/or connections
        update_possible_poses(connections, possible_poses, idx_jointA, idx_jointB, candidates, keypoints_number);
    }
}

static inline std::vector<TwoJointsConnection> form_most_promising_joint_connections(const size_t keypoints_number, const std::vector<cv::Mat> &pafs,
                                                                                     const std::pair<int, int> (&limb_ids_paf)[N_PAF_LIMB_PAIRS],
                                                                                     const size_t limb_idx, const size_t njointsA, const size_t njointsB,
                                                                                     const std::vector<Peak> &candidates_for_jointA,
                                                                                     const std::vector<Peak> &candidates_for_jointB,
                                                                                     const float mid_points_score_threshold,
                                                                                     const float found_mid_points_ratio_threshold)
{
        // Get two PAFs - the ones from the front half of the array which correspond to the two we already have
        const int map_idx_offset = keypoints_number + 1;
        std::pair<cv::Mat, cv::Mat> score_mid = {
            pafs[limb_ids_paf[limb_idx].first - map_idx_offset], // PAF at index [first limb's index - (nkeypoints + 1)]
            pafs[limb_ids_paf[limb_idx].second - map_idx_offset] // PAF at index [second limb's index - (nkeypoints + 1)]
        };

        // Create a temporary list of joint connections - these are possible limbs
        std::vector<TwoJointsConnection> temp_joint_connections = create_temp_joint_connections(njointsA, njointsB, candidates_for_jointA, candidates_for_jointB, score_mid, pafs, mid_points_score_threshold, found_mid_points_ratio_threshold);

        // Sort the limbs by their line integral across the PAFs - the ones with the highest score are the ones that
        // are most likely to be real and thereby incorporated into poses
        if (!temp_joint_connections.empty())
        {
            std::sort(temp_joint_connections.begin(), temp_joint_connections.end(), [](const TwoJointsConnection& a, const TwoJointsConnection& b) {
                return (a.score > b.score);
            });
        }

        // Take the most promising joint connections (limbs) from temp_joint_connections
        return take_most_promising_joint_connections(temp_joint_connections, njointsA, njointsB, candidates_for_jointA, candidates_for_jointB);
}

static inline void maybe_create_possible_pose(std::vector<HumanPoseByPeaksIndices> &possible_poses, const size_t idx_joint,
                                              const Peak &candidate_peak, const size_t keypoints_number)
{
    // For each possible human skeleton (pose)
    for (size_t j = 0; j < possible_poses.size(); j++)
    {
        // If this pose's joint of interest is our candidate peak, we won't create another pose with this peak
        if (possible_poses[j].peaks_indices[idx_joint] == candidate_peak.id)
        {
            return;
        }
    }

    // If no skeletons already have our candidate peak as their joint of interest,
    // create another possible skeleton, with this candidate peak as its whatever joint
    HumanPoseByPeaksIndices person_keypoints(keypoints_number);
    person_keypoints.peaks_indices[idx_joint] = candidate_peak.id;
    person_keypoints.njoints = 1;
    person_keypoints.score = candidate_peak.score;
    possible_poses.push_back(person_keypoints);
}

static inline void create_possible_poses_from_peaks(const size_t npeaks, std::vector<HumanPoseByPeaksIndices> &possible_poses, const int idx_joint,
                                                    const std::vector<Peak> &candidate_peaks_for_joint, const size_t keypoints_number)
{
    // For each peak (possible real joint), maybe create a new possible pose
    for (size_t i = 0; i < npeaks; i++)
    {
        maybe_create_possible_pose(possible_poses, idx_joint, candidate_peaks_for_joint[i], keypoints_number);
    }
}

static inline void maybe_create_possible_poses_from_peaks(const size_t njointsA, const size_t njointsB, std::vector<HumanPoseByPeaksIndices> &possible_poses,
                                                          const int idx_jointA, const int idx_jointB, const std::vector<Peak> &candidates_for_jointA,
                                                          const std::vector<Peak> &candidates_for_jointB, const size_t keypoints_number)
{
    if ((njointsA == 0) && (njointsB == 0))
    {
        // If there are no joints in either heat map, just give up
        return;
    }
    else if (njointsA == 0)
    {
        // If there are no joints in heat map A, create possible poses from just B
        // Even though there are no combinations (limbs) to be made, we may still incorporate some of B's joints into skeletons
        create_possible_poses_from_peaks(njointsB, possible_poses, idx_jointB, candidates_for_jointB, keypoints_number);
    }
    else if (njointsB == 0)
    {
        // If there are no joints in heat map B, create possible poses from just A
        // Even though there are no combinations (limbs) to be made, we may still incorporate some of A's joints into skeletons
        create_possible_poses_from_peaks(njointsA, possible_poses, idx_jointA, candidates_for_jointA, keypoints_number);
    }
}

/**
 * Fills in `possible_poses`.
 *
 * We look at two heatmaps - A and B. These two heatmaps have all of joint type A (say, neck) and
 * all of joint type B (say right shoulder). The heatmaps we look at are found by looking at `limb_ids_heatmap[limb_idx]`.
 *
 * We create a giant set of all the most likely connections between possible joints in A to possible joints in B,
 * such that each possible joint in A can only be connected to at most one of the possible joints in B (and vice versa),
 * and the weight of this bipartite graph is maximized.
 *
 * The weight of each of these connections is determined by looking at the PAFs - specifically,
 * we approximate a line integral from one joint candidate to the other over the right PAF.
 * Because PAFs are vector fields made so that the vectors point along limbs, if A and B are really connected
 * (such as neck and right shoulder), we will likely get a good score when computing the line integral, as we
 * will be traveling along the vector field. If on the other hand, the joints are not connected, we will likely
 * travel over the vector field wherever there are maginitude 0 vectors or where we are perpendicular to the vector field.
 *
 * @param keypoints_number: The number of keypoints in a typical HumanPose
 * @param pafs: The PAFs
 * @param limb_ids_paf: The array of PAF joint IDs
 * @param limb_ids_heatmap: The array of joint IDs that make up the limbs
 * @param all_peaks: All the peaks by heatmap
 * @param mid_points_score_threshold:
 * @param found_mid_points_ratio_threshold:
 * @param candidates: All the candidate peaks for forming into skeletons
 * @param limb_idx: The particular limb (used as an index into the limb arrays)
 * @param possible_poses: We fill this vector with possible poses
 */
static inline void create_candidate_poses_by_limb(const size_t keypoints_number, const std::vector<cv::Mat> &pafs, const std::pair<int, int> (&limb_ids_paf)[N_PAF_LIMB_PAIRS],
                                                  const std::pair<int, int> (&limb_ids_heatmap)[N_HEATMAP_LIMB_PAIRS], const std::vector<std::vector<Peak>> &all_peaks,
                                                  const float mid_points_score_threshold, const float found_mid_points_ratio_threshold,
                                                  const std::vector<Peak> &candidates, const size_t limb_idx, std::vector<HumanPoseByPeaksIndices> &possible_poses)
{
    // A 'limb' is made up of two joints.
    // Get the heatmap indices for those two joints (the heatmaps that contain the peaks that represent those joints).
    const int idx_jointA = limb_ids_heatmap[limb_idx].first - 1;
    const int idx_jointB = limb_ids_heatmap[limb_idx].second - 1;

    // Get all the peaks (possible keypoints/joint locations) for the two joints that make up this limb
    const std::vector<Peak> &candidates_for_jointA = all_peaks[idx_jointA];
    const std::vector<Peak> &candidates_for_jointB = all_peaks[idx_jointB];

    // Get the number of joints in the two heatmaps
    const size_t njointsA = candidates_for_jointA.size();
    const size_t njointsB = candidates_for_jointB.size();

    if ((njointsA == 0) || (njointsB == 0))
    {
        maybe_create_possible_poses_from_peaks(njointsA, njointsB, possible_poses, idx_jointA, idx_jointB, candidates_for_jointA, candidates_for_jointB, keypoints_number);
    }
    else
    {
        std::vector<TwoJointsConnection> connections = form_most_promising_joint_connections(keypoints_number, pafs, limb_ids_paf, limb_idx, njointsA, njointsB, candidates_for_jointA, candidates_for_jointB, mid_points_score_threshold, found_mid_points_ratio_threshold);
        create_poses_from_connections(connections, limb_idx, possible_poses, keypoints_number, idx_jointA, idx_jointB, candidates);
    }
}

void group_peaks_to_poses(const std::vector<std::vector<Peak>> &all_peaks, const std::vector<cv::Mat> &pafs, const size_t keypoints_number,
                          const float mid_points_score_threshold, const float found_mid_points_ratio_threshold, const int min_joints_number,
                          const float min_subset_score, std::vector<HumanPose> &poses)
{
    // These are the pairs of heat maps which go together for a human
    static const std::pair<int, int> limb_ids_heatmap[N_HEATMAP_LIMB_PAIRS] = {
        {2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10}, {10, 11}, {2, 12}, {12, 13}, {13, 14},
        {2, 1}, {1, 15}, {15, 17}, {1, 16}, {16, 18}, {3, 17}, {6, 18}
    };

    // These are the pairs of PAFs that form limbs; what about all the other PAFs? ¯\_(ツ)_/¯
    static const std::pair<int, int> limb_ids_paf[N_PAF_LIMB_PAIRS] = {
        {31, 32}, {39, 40}, {33, 34}, {35, 36}, {41, 42}, {43, 44}, {19, 20}, {21, 22}, {23, 24}, {25, 26},
        {27, 28}, {29, 30}, {47, 48}, {49, 50}, {53, 54}, {51, 52}, {55, 56}, {37, 38}, {45, 46}
    };

    // To start with, all heat map peaks are candidate keypoints
    std::vector<Peak> candidates;
    for (const auto &peaks : all_peaks)
    {
         candidates.insert(candidates.end(), peaks.begin(), peaks.end());
    }

    // For each PAF pair that makes up a limb, use it to create a list of candidate skeletons
    std::vector<HumanPoseByPeaksIndices> possible_poses(0, HumanPoseByPeaksIndices(keypoints_number));
    for (size_t k = 0; k < util::array_size(limb_ids_paf); k++)
    {
        create_candidate_poses_by_limb(keypoints_number, pafs, limb_ids_paf, limb_ids_heatmap, all_peaks, mid_points_score_threshold, found_mid_points_ratio_threshold, candidates, k, possible_poses);
    }

    // For each subset of keypoints in the vector of HumanPoseByPeaksIndices, check if we can form it
    // into a real HumanPose object, and if we can, do so.
    for (const auto &possible_pose : possible_poses)
    {
        // If this subset of keypoints has too few keypoints in it or if its score over njoints is less than
        // min_subset_score, we exclude it from the poses.
        if ((possible_pose.njoints < min_joints_number) || ((possible_pose.score / possible_pose.njoints) < min_subset_score))
        {
            continue;
        }

        // Create a HumanPose out of the subset of joints
        int position = -1;
        HumanPose pose(std::vector<cv::Point2f>(keypoints_number, cv::Point2f(-1.0f, -1.0f)), possible_pose.score * std::max(0, possible_pose.njoints - 1));
        for (const auto &peak_idx : possible_pose.peaks_indices)
        {
            position++;
            if (peak_idx >= 0)
            {
                pose.keypoints[position] = candidates[peak_idx].pos;
                pose.keypoints[position].x += 0.5;
                pose.keypoints[position].y += 0.5;
            }
        }
        poses.push_back(pose);
    }
}

} // namespace peak
} // namespace pose
