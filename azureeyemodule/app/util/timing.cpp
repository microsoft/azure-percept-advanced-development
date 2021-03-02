// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>

// Local includes
#include "timing.hpp"

namespace ourtime
{

std::string get_timestamp()
{
    time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

void Timer::reset()
{
    this->begin = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const
{
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - this->begin).count();
}

double Timer::elapsed_ms() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - this->begin).count();
}

} // namespace ourtime
