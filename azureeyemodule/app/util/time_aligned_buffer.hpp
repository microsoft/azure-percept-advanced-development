// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * This module represents a framebuffer that can be used for storing timestamped frames.
 * We use this for time-aligning frames with neural network inferences.
 */
#pragma once

// Standard library includes
#include <string>
#include <tuple>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>

// Local includes
#include "helper.hpp"

namespace timebuf {

/** A tuple of cv::Mat and timestamp for that frame. */
using timestamped_frame_t = std::tuple<cv::Mat, int64_t>;

class TimeAlignedBuffer
{
public:
    /** Constructor. Takes a default value to return until we get frames in the buffer. */
    TimeAlignedBuffer(const cv::Mat &default_item);

    /** Copies the given frame and timestamp into the buffer, overwriting an old one if the we end up wrapping. */
    void put(const timestamped_frame_t &frame_and_ts);

    /** Removes the best matching frame and all older ones and returns them as a vector. If no frames in buffer, we return the default one or the last one we returned. */
    std::vector<cv::Mat> get_best_match_and_older(int64_t timestamp);

    /** Returns the current number of items in the buffer. */
    size_t size() const;

private:
    /** The index to write something to. */
    size_t index = 0;

    /** Number of timestamped frames that we can currently have in the buffer. Increases if we need more capacity. */
    size_t n_timestamped_frames;

    /** The frame we return if there are no frames to return. */
    cv::Mat default_value;

    /** Circular buffer of frames with their timestamps. */
    std::vector<timestamped_frame_t> timestamped_frames;
};

} // namespace timebuf