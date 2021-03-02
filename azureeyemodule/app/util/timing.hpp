// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <chrono>
#include <future>
#include <string>
#include <thread>

namespace ourtime {

/** Return a string timestamp */
std::string get_timestamp();

/** A simple timer class */
class Timer
{
public:
    Timer() : begin(std::chrono::high_resolution_clock::now()) {}

    /** Reset the timer. */
    void reset();

    /** Return how many seconds have elapsed since either construction or the last `reset()` call. */
    double elapsed() const;

    /** Return how many milliseconds have elapsed since either construction or the last `reset()` call. */
    double elapsed_ms() const;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> begin;
};

} // namespace ourtime
