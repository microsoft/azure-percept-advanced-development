// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <chrono>
#include <thread>

// Local includes
#include "framebuffer.hpp"
#include "resolution.hpp"
#include "../util/circular_buffer.hpp"
#include "../util/helper.hpp"

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

namespace rtsp {

FrameBuffer::FrameBuffer(size_t max_length, int fps)
    : circular_buffer(max_length), cached_frame(DEFAULT_HEIGHT, DEFAULT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
      fps(fps), fps_thread( std::thread([this]{this->periodically_update_frame();}) )
{
}

FrameBuffer::~FrameBuffer()
{
    // Signal the thread we are terminating
    this->shut_down = true;

    // Wait for it to join. This could take up to 1/FPS seconds.
    this->fps_thread.join();
}

cv::Mat FrameBuffer::get(const Resolution &resolution)
{
    // Readers may have to wait for the fps_update thread to update the latest frame,
    // but that shouldn't take long.

    cv::Mat ret;

    // Take the mutex in order to read the cached frame (which may
    // be in the middle of being updated by the FPS thread)
    this->cached_frame_mutex.lock();
    this->cached_frame.copyTo(ret);
    this->cached_frame_mutex.unlock();

    // Determine the resolution the caller wants
    int desired_height;
    int desired_width;
    std::tie(desired_height, desired_width) = get_height_and_width(resolution);

    // What resolution is the image that we got?
    int ret_height = ret.size().height;
    int ret_width = ret.size().width;

    // If not the right resolution, we should update.
    if ((ret_height != desired_height) || (ret_width != desired_width))
    {
        cv::resize(ret, ret, cv::Size(desired_width, desired_height));
    }

    return ret;
}

void FrameBuffer::put(const cv::Mat &frame)
{
    // Writers block until they put a frame into the buffer, but nobody actually blocks reading
    // from the buffer in this design, so we should always succeed without waiting.
    // Unless in the future we re-use this class for some other purpose or have multiple writers.
    this->circular_buffer.put(frame);
}

void FrameBuffer::periodically_update_frame()
{
    while (!this->shut_down)
    {
        // Try to grab the next frame from the buffer, but it might be empty,
        // or it might be in the middle of a put(). Either way, we don't have
        // time to wait on this thread, as we may be running at a high FPS.
        // We'll just catch it again next time.
        cv::Mat frame;
        bool got = this->circular_buffer.get_no_wait(frame);

        // We do need to grab the cached_frame lock though.
        if (got)
        {
            this->cached_frame_mutex.lock();
            this->cached_frame = frame.clone();
            this->cached_frame_mutex.unlock();
        }

        // Sleep for (1.0 / fps) seconds.
        assert(this->fps != 0);
        double one_over_fps = 1.0 / (double)this->fps;
        double sleep_duration_ms = std::ceil(one_over_fps * 1000.0);
        std::this_thread::sleep_for(std::chrono::milliseconds((int64_t)sleep_duration_ms));
    }
}

size_t FrameBuffer::room() const
{
    auto capacity = this->circular_buffer.capacity();
    auto size = this->circular_buffer.size_no_wait();
    if (size > capacity)
    {
        // We won the lottery! Someone happened to be updating the circular buffer
        // in just the right way at just the right time so that calling size_no_wait()
        // returned something bogus.
        // Safest thing to do in this most unlikely of circumstances is to return 0.
        util::log_debug("FrameBuffer::room() got a bogus value from circular_buffer.size_no_wait(): " + std::to_string(size) + " when capacity is " + std::to_string(capacity));
        return 0;
    }
    else
    {
        return capacity - size;
    }
}

} // namespace rtsp