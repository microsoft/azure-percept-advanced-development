// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Local includes
#include "resolution.hpp"
#include "../util/helper.hpp"

// Standard library includes
#include <assert.h>
#include <string>

namespace rtsp {

std::string resolution_to_string(const Resolution &resolution)
{
    switch (resolution)
    {
        case Resolution::NATIVE:
            return "native";
        case Resolution::HD1080P:
            return "1080p";
        case Resolution::HD720P:
            return "720p";
        default:
            util::log_error("Unrecognized resolution type when trying to convert to string.");
            assert(false);
            return "UNKNOWN";
    }
}

bool is_valid_resolution(const std::string &resolution)
{
    if (resolution == "native")
    {
        return true;
    }
    else if (resolution == "1080p")
    {
        return true;
    }
    else if (resolution == "720p")
    {
        return true;
    }
    else
    {
        return false;
    }
}

Resolution resolution_string_to_enum(const std::string &resolution)
{
    if (resolution == "native")
    {
        return Resolution::NATIVE;
    }
    else if (resolution == "1080p")
    {
        return Resolution::HD1080P;
    }
    else if (resolution == "720p")
    {
        return Resolution::HD720P;
    }
    else
    {
        throw std::invalid_argument("Invalid resolution string.");
    }
}

std::tuple<int, int> get_height_and_width(const Resolution &resolution)
{
    switch (resolution)
    {
        case Resolution::NATIVE:
            return {DEFAULT_HEIGHT, DEFAULT_WIDTH};
        case Resolution::HD1080P:
            return {1080, 1920};
        case Resolution::HD720P:
            return {720, 1280};
        default:
            util::log_error("Invalid resolution when trying to get height and width.");
            assert(false);
            return {DEFAULT_HEIGHT, DEFAULT_WIDTH};
    }
}

} // namespace rtsp