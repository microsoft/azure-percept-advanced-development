// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "helper.hpp"

namespace util {

static bool verbose_logging = false;

bool file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

void log_error(const std::string &str)
{
    time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::put_time(std::localtime(&t), "%Y-%m-%d %X") << " ERROR: " << str << std::endl;
}

void log_info(const std::string &str)
{
    time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::put_time(std::localtime(&t), "%Y-%m-%d %X") << " " << str << std::endl;
}

void log_debug(const std::string &str)
{
    if (verbose_logging)
    {
        time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::put_time(std::localtime(&t), "%Y-%m-%d %X") << " " << str << std::endl;
    }
}

std::string get_label(int index, const std::vector<std::string> &class_list)
{
    if ((index >= 0) && (static_cast<size_t>(index) < class_list.size()))
    {
        return class_list[index];
    }
    else
    {
        return std::to_string(index);
    }
}

void put_text(const cv::Mat &rgb, std::string text)
{
    cv::putText(rgb,
        text,
        cv::Point(300, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        5);

    cv::putText(rgb,
        text,
        cv::Point(300, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255, 255, 255),
        2);
}

void set_logging(bool data)
{
    verbose_logging = data;
    log_info("verbose_logging: " + std::to_string(verbose_logging));
}

std::string to_hex_string(int i)
{
    std::stringstream ss;
    ss << std::hex << i;
    return ss.str();
}

std::string to_lower(std::string str)
{
    // convert string to back to lower case
    std::for_each(str.begin(), str.end(), [](char& c)
        {
            c = ::tolower(c);
        });

    return str;
}

std::string to_size_string(cv::Mat& mat)
{
    std::string str = std::string("");

    for (int i = 0; i < mat.dims; ++i)
    {
        // Append x234 (for example), but only if it is not the first dim
        str.append(i ? "x" : "").append(std::to_string(mat.size[i]));
    }

    return str;
}

std::string to_string_with_precision(float f, int precision)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << f;
    return ss.str();
}

int run_command(std::string command)
{
    log_info(command);
    return system(command.c_str());
}

bool search_keyword_in_file(std::string keyword, std::string filename)
{
    std::ifstream file(filename);

    if (file.is_open())
    {
        std::string line;

        while (getline(file, line))
        {
            if (std::string::npos != line.find(keyword))
            {
                return true;
            }
        }

        file.close();
    }

    return false;
}

std::string timestamp_to_string(int64_t ts)
{
    // Compose date, hour, minute, and second
    static char buf[100];
    std::time_t ts_tmp = (int64_t)(ts / 1E9);
    struct std::tm *time = gmtime(&ts_tmp);
    std::strftime(buf, sizeof(buf), "%F %I:%M:%S%p ", time);
    std::string first_half = std::string(buf);

    // Compose millisecond, microsecond, and nanosecond
    //
    // Get the number of seconds we just used in the first half of the timestamp
    auto ts_as_duration = std::chrono::duration<int64_t, std::nano>(ts);
    auto ts_as_timepoint = std::chrono::time_point<std::chrono::system_clock>(ts_as_duration);
    auto time_since_epoch_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(ts_as_timepoint.time_since_epoch());

    // Convert that to nanosecond resolution (just zeros from seconds on)
    auto seconds_portion_in_nanoseconds_resolution = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch_in_seconds).count();

    // Get the remaining number of nanoseconds after we subtract the nanoseconds it took to get the hour, minute, second
    assert(ts >= seconds_portion_in_nanoseconds_resolution);
    int64_t ns_remaining = ts - seconds_portion_in_nanoseconds_resolution;

    // Turn this number of nanoseconds into milliseconds, microseconds, and finally, remaining nanoseconds
    auto ms = (int64_t)(ns_remaining / 1E6);
    ns_remaining -= (int64_t)(ms * 1E6);

    auto us = (int64_t)(ns_remaining / 1E3);
    ns_remaining -= (int64_t)(us * 1E3);

    auto ns = ns_remaining;

    // Compose the second half of the string
    std::string second_half = std::to_string(ms) + "ms " + std::to_string(us) + "us " + std::to_string(ns) + "ns";
    return first_half + second_half;
}

void version()
{
    system("stat -c 'Inference App Version: %y' /app/inference | cut -c -33");
}

std::vector<std::string> splice_comma_separated_list(const std::string &list_string)
{
    std::vector<std::string> result;
    std::stringstream ss(list_string);

    while (ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        result.push_back(substr);
    }

    return result;
}

AdaptiveLogger::AdaptiveLogger()
    : last_timestamp(std::chrono::milliseconds(0)), threshold(std::chrono::milliseconds(0))
{
}

void AdaptiveLogger::log_info(const std::string &msg)
{
    bool log_this_msg = this->adapt();
    if (log_this_msg)
    {
        util::log_info(msg);
    }
}

void AdaptiveLogger::log_error(const std::string &msg)
{
    bool log_this_msg = this->adapt();
    if (log_this_msg)
    {
        util::log_error(msg);
    }
}

void AdaptiveLogger::log_debug(const std::string &msg)
{
    bool log_this_msg = this->adapt();
    if (log_this_msg)
    {
        util::log_debug(msg);
    }
}

bool AdaptiveLogger::adapt()
{
    bool should_log;

    // Get current timestamp
    std::chrono::milliseconds timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    // Compare against the last successful timestamp
    std::chrono::milliseconds ms_since_last_log = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - this->last_timestamp);

    // If the last time we logged was longer ago than threshold, we can log again.
    if (ms_since_last_log > this->threshold)
    {
        this->threshold = std::chrono::duration_cast<std::chrono::milliseconds>(1.5 * this->threshold);
        should_log = true;
    }
    else
    {
        should_log = false;
    }

    // Clamp the threshold to min and max
    if (this->threshold < std::chrono::milliseconds(5))
    {
        this->threshold = std::chrono::milliseconds(5);
    }
    else if (this->threshold > this->maximum)
    {
        this->threshold = this->maximum;
    }

    // Update last successful timestamp
    if (should_log)
    {
        this->last_timestamp = timestamp;
        util::log_info("Logging a verbose message. Next time allowed to log will be in " + std::to_string(this->threshold.count()) + " ms.");
    }

    return should_log;
}

} // namespace util
