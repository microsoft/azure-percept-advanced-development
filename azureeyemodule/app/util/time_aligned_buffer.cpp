// Standard library includes
#include <vector>

// Local includes
#include "time_aligned_buffer.hpp"

namespace timebuf {


TimeAlignedBuffer::TimeAlignedBuffer(const cv::Mat &default_item)
    : n_timestamped_frames(10), default_value(default_item)
{
    // Nothing to do
}

void TimeAlignedBuffer::put(const timestamped_frame_t &frame_and_ts)
{
    // Add the item
    if (this->index >= this->timestamped_frames.size())
    {
        // Add a new slot
        this->timestamped_frames.push_back(frame_and_ts);
    }
    else
    {
        // Overwrite an old item's slot
        this->timestamped_frames.at(this->index) = frame_and_ts;
    }

    this->index += 1;
    if (this->index >= this->n_timestamped_frames)
    {
        this->index = 0;
    }
}

size_t TimeAlignedBuffer::size() const
{
    return this->timestamped_frames.size();
}

std::vector<cv::Mat> TimeAlignedBuffer::get_best_match_and_older(int64_t timestamp)
{
    // If there is nothing in the buffer yet, let's return a default value.
    if (this->timestamped_frames.size() == 0)
    {
        #ifdef DEBUG_TIME_ALIGNMENT
            util::log_debug("New Inference: No frames in buffer. Sending cached frame.");
        #endif
        return this->default_value;
    }

    cv::Mat oldest_frame;
    int64_t oldest_ts = LLONG_MAX;

    int64_t best_match_ts = 0;
    int64_t smallest_time_difference = LLONG_MAX;

    // Sanity check
    assert(this->timestamped_frames.size() != 0);

    // Search through the circular buffer for the oldest timestamp and the best matching timestamp.
    for (const auto &tup : this->timestamped_frames)
    {
        auto tup_ts = std::get<1>(tup);
        if (tup_ts < oldest_ts)
        {
            std::tie(oldest_frame, oldest_ts) = tup;
        }

        // Also find the best match
        auto timedelta = (tup_ts > timestamp) ? (tup_ts - timestamp) : (timestamp - tup_ts);
        if (timedelta < smallest_time_difference)
        {
           best_match_ts = std::get<1>(tup);
           smallest_time_difference = timedelta;
        }
    }

    // Sanity check
    assert(oldest_ts != LLONG_MAX);
    assert(best_match_ts != 0);
    assert(smallest_time_difference != LLONG_MAX);
    #ifdef DEBUG_TIME_ALIGNMENT
        util::log_debug("New Inference: Matched " + util::timestamp_to_string(timestamp) + " with " + util::timestamp_to_string(best_match_ts));
    #endif

    if (oldest_ts > timestamp)
    {
        #ifdef DEBUG_TIME_ALIGNMENT
            util::log_debug("New Inference: oldest frame is not old enough! We will store " + std::to_string(this->n_timestamped_frames * 2) + " frames now.");
        #endif

        // If oldest frame occurs after this
        // frame's timestamp, we are overwriting our buffer frames before the network can inference even a single time.
        // Therefore the buffer is not large enough and needs to be resized.
        this->n_timestamped_frames *= 2;

        // Use this frame as the best one, but don't remove it, since it doesn't really match.
        assert(best_match_ts == oldest_ts);
        return std::vector<cv::Mat>{oldest_frame};
    }
    else
    {
        // If the oldest frame occurs before this inference, then the frame we are inferencing on
        // is somewhere in our buffer. Search through to find it and all older frames.
        std::vector<cv::Mat> best_and_older;
        std::vector<size_t> indices_to_erase;
        for (size_t i = 0; i < this->timestamped_frames.size(); i++)
        {
            const auto &tup = this->timestamped_frames.at(i);
            if (std::get<1>(tup) <= best_match_ts)
            {
                indices_to_erase.push_back(i);
                best_and_older.push_back(std::get<0>(tup));
            }
        }

        // Update the default value
        this->default_value = best_and_older.back();

        // Remove all the old frames.
        assert(best_and_older.size() > 0);
        assert(indices_to_erase.size() > 0);
        auto start = indices_to_erase.at(0);
        auto end = indices_to_erase.back() + 1;
        this->timestamped_frames.erase(this->timestamped_frames.begin() + start, this->timestamped_frames.begin() + end);

        // We need to decrement index by an amount equal
        // to the number of indices we removed that are less than it.
        size_t amount = 0;
        for (auto i : indices_to_erase)
        {
            if (i < this->index)
            {
                amount++;
            }
        }
        assert(amount <= this->index);
        this->index -= amount;

        #ifdef DEBUG_TIME_ALIGNMENT
            util::log_debug("New Inference: Found " + std::to_string(best_and_older.size()) + " frames. Now have " + std::to_string(this->size()) + " frames left.");
        #endif

        return best_and_older;
    }
}

} // namespace timebuf