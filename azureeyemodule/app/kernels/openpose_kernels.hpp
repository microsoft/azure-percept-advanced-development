// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <vector>

// Third party includes
#include <opencv2/core/core.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/garg.hpp>       // IStreamSource
#include <opencv2/gapi/gkernel.hpp>    // GKernelPackage
#include <opencv2/gapi/gstreaming.hpp> // GOptRunArgsP
#include <opencv2/gapi/cpu/gcpukernel.hpp> // GAPI_OCV_KERNEL macro

// Local includes
#include "../openpose/human_pose.hpp"
#include "../openpose/peak.hpp"

namespace cv {
namespace gapi {
namespace streaming {

using GPoses = cv::GArray<pose::HumanPose>;

/**
 * A parallel implementation on the CPU for finding all the peaks in all the heat maps.
 */
class FindPeaksBody: public cv::ParallelLoopBody
{
public:
    /**
     * @param heat_maps: All the heat maps
     * @param min_peaks_distance: Peaks in heat maps must be at least this far away in order to be counted as separate.
     * @param peaks_from_heatmap: We will fill this with the peaks. Each item in the outer vector
     *                            is a list of peaks (possible key points) for a corresponding heat map.
     */
    FindPeaksBody(const std::vector<cv::Mat> &heat_maps, float min_peaks_distance, std::vector<std::vector<pose::peak::Peak>> &peaks_from_heatmap)
        : heat_maps(heat_maps), min_peaks_distance(min_peaks_distance), peaks_from_heatmap(peaks_from_heatmap)
    {
        // Nothing to do
    }

    virtual void operator()(const cv::Range &range) const
    {
        for (int i = range.start; i < range.end; i++)
        {
            pose::peak::find_peaks(heat_maps, min_peaks_distance, peaks_from_heatmap, i);
        }
    }

private:
    /** The heat maps (key points) */
    const std::vector<cv::Mat> &heat_maps;
    /** Peaks in heat maps must be at least this far away in order to be counted as separate. */
    float min_peaks_distance;
    /** All peaks in a heat map, for each heat map */
    std::vector<std::vector<pose::peak::Peak>> &peaks_from_heatmap;
};

/**
 * Parse OpenPose outputs.
 *
 * This defines a graph operation called GParseOpenPose. When you call this op's ::on method,
 * it executes the kernel's run function.
 */
G_API_OP(GParseOpenPose, <GPoses(GMat, GMat, GOpaque<Size>)>, "org.opencv.dnn.parse_open_pose")
{
    static GArrayDesc outMeta(const GMatDesc&, const GMatDesc&, const GOpaqueDesc&)
    {
        return empty_array_desc();
    }
};

/**
 * @brief Parses output of OpenPose network.
 */
GAPI_EXPORTS GPoses parse_open_pose(const GMat &pafs, const GMat &keys, const GOpaque<Size> &in_sz);

/**
 * Fills `poses` with the skeletons - one for each person in the image.
 *
 * @param heat_maps: The heat maps in vector of heat maps form.
 * @param pafs: The part affinity fields in vector of PAFs form.
 * @param poses: A vector of poses we will fill.
 * @param min_peaks_distance: The minimum distance between peaks (keypoints) for us to consider them different peaks.
 * @param keypoints_number: The number of keypoints in a skeleton.
 * @param mid_points_score_threshold:
 * @param found_mid_points_ratio_threshold:
 * @param min_joints_number: The minimum number of joints to be considered a pose.
 * @param min_subset_score:
 */
void extract_poses(const std::vector<cv::Mat> &heat_maps, const std::vector<cv::Mat> &pafs, std::vector<pose::HumanPose> &poses,
                   const float min_peaks_distance, const size_t keypoints_number, const float mid_points_score_threshold,
                   const float found_mid_points_ratio_threshold, const int min_joints_number, const float min_subset_score);

/**
 * TODO: Document me
 */
void correct_coordinates(std::vector<pose::HumanPose> &poses, const cv::Size &feature_map_size, const cv::Size &image_size, int stride, int upsample_ratio);

/**
 * Kernel implementation for OpenPose parsing.
 *
 * @param in_pafs: A rank 4 tensor of shape [1, 38, 32, 57]
 * @param in_keys: A rank 4 tensor of shape [1, 19, 32, 57]
 * @param in_size: A Size type detailing the dimensions of the image.
 * @param out_poses: A vector of Poses that we return.
 */
GAPI_OCV_KERNEL(GOCVParseOpenPose, GParseOpenPose)
{
    static void run(const Mat &in_pafs, const Mat &in_keys, const Size &in_size, std::vector<pose::HumanPose> &out_poses)
    {
        // Clear the poses to make room for this time.
        out_poses.clear();

#ifndef NDEBUG
        /** Dimensions we expect in_pafs to be */
        int paf_sizes[] = {1, 38, 32, 57};

        /** Dimensions we expect in_keys to be */
        int key_sizes[] = {1, 19, 32, 57};

        // Sanity check that dimensions are what we expect for OpenPose
        assert(in_pafs.size[0] == paf_sizes[0]);
        assert(in_pafs.size[1] == paf_sizes[1]);
        assert(in_pafs.size[2] == paf_sizes[2]);
        assert(in_pafs.size[3] == paf_sizes[3]);
        assert(in_keys.size[0] == key_sizes[0]);
        assert(in_keys.size[1] == key_sizes[1]);
        assert(in_keys.size[2] == key_sizes[2]);
        assert(in_keys.size[3] == key_sizes[3]);
#endif //NDEBUG

        // Hyper parameters for the open pose model parser

        /** TODO: Document me */
        static const int stride = 8;

        /** Candidate peaks must be at least this far away from one another or they are collapsed into one */
        static const float min_peaks_distance = 3.0f;

        /** TODO: Document me */
        static const float mid_points_score_threshold = 0.05f;

        /** TODO: Document me */
        static const float found_mid_points_ratio_threshold = 0.8f;

        /** TODO: Document me */
        static const float min_subset_score = 0.2f;

        /** TODO: Document me */
        static const int upsample_ratio = 4;

        /** The minimum number of joints in a skeleton - skeleton candidates with fewer points are discarded. */
        static const int min_joints_number = 3;

        /** The number of keypoints in a complete skeleton. */
        static const int nkeypoints = 18;

        // Copy the data from the const-qualified args (the GAPI requires them to be const as far as I can tell)
        cv::Mat in_keys_copy(in_keys);
        cv::Mat in_pafs_copy(in_pafs);

        // Reshape in_keys from [1, 19, 32, 57] to [19, 32, 57]
        int ksizes[] = {in_keys.size[1], in_keys.size[2], in_keys.size[3]};
        cv::Mat reshaped_heatmaps(3, ksizes, in_keys.type(), reinterpret_cast<void *>(in_keys_copy.ptr<float>(0)));

        // Reshape in_pafs from [1, 38, 32, 57] to [38, 32, 57]
        const int psizes[] = {in_pafs.size[1], in_pafs.size[2], in_pafs.size[3]};
        cv::Mat reshaped_pafs(3, psizes, in_pafs.type(), reinterpret_cast<void *>(in_pafs_copy.ptr<float>(0)));

#ifndef NDEBUG
        // Sanity check that the dimensions are what we expect now
        assert(reshaped_heatmaps.size[0] == in_keys.size[1]);
        assert(reshaped_heatmaps.size[1] == in_keys.size[2]);
        assert(reshaped_heatmaps.size[2] == in_keys.size[3]);
        assert(reshaped_pafs.size[0] == in_pafs.size[1]);
        assert(reshaped_pafs.size[1] == in_pafs.size[2]);
        assert(reshaped_pafs.size[2] == in_pafs.size[3]);
#endif

        // Create a much easier-to-use datastructure out of this tensor - a vector of 2D matrices (OpenCV does not do tensors well)
        const int ksizes2d[] = {in_keys.size[2], in_keys.size[3]};
        std::vector<cv::Mat> heatmaps(in_keys.size[1]);
        for (size_t i = 0; i < heatmaps.size(); i++)
        {
            // Pull out the next heatmap
            cv::Range indices[] = { cv::Range(i, i + 1), cv::Range::all(), cv::Range::all() };
            auto tmp = cv::Mat(reshaped_heatmaps(indices));

            // Remove its first dimension, so now it is a 2D Mat
            heatmaps[i] = cv::Mat(2, ksizes2d, in_keys.type(), reinterpret_cast<void *>(tmp.ptr<float>(0)));

            // I don't know about this:
            cv::resize(heatmaps[i], heatmaps[i], cv::Size(), upsample_ratio, upsample_ratio, cv::INTER_CUBIC);
        }

        // Do the same for the PAFs
        const int psizes2d[] = {in_pafs.size[2], in_pafs.size[3]};
        std::vector<cv::Mat> pafs(in_pafs.size[1]);
        for (size_t i = 0; i < pafs.size(); i++)
        {
            // Pull out the next PAF
            cv::Range indices[] = { cv::Range(i, i + 1), cv::Range::all(), cv::Range::all() };
            auto tmp = cv::Mat(reshaped_pafs(indices));

            // Remove its first dimension, so now it is a 2D Mat
            pafs[i] = cv::Mat(2, psizes2d, in_pafs.type(), reinterpret_cast<void *>(tmp.ptr<float>(0)));

            // I don't know about this:
            cv::resize(pafs[i], pafs[i], cv::Size(), upsample_ratio, upsample_ratio, cv::INTER_CUBIC);
        }

        // Now extract all the poses out of the data
        extract_poses(heatmaps, pafs, out_poses, min_peaks_distance, nkeypoints, mid_points_score_threshold, found_mid_points_ratio_threshold, min_joints_number, min_subset_score);

        // Correct the coordinates of the poses
        correct_coordinates(out_poses, heatmaps[0].size(), in_size, stride, upsample_ratio);
    }
};

} // namespace streaming
} // namespace gapi
} // namespace cv
