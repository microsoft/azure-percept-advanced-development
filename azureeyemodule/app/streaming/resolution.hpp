// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

// Standard library includes
#include <string>
#include <tuple>

namespace rtsp {

/** Default height of the RTSP images. */
const int DEFAULT_HEIGHT = 616;

/** Default width of the RTSP images. */
const int DEFAULT_WIDTH = 816;

/** The different resolutions we can use. */
enum class Resolution {
   NATIVE,
   HD1080P,
   HD720P,
};

/** Returns true if the given resolution string is valid. False if not. */
bool is_valid_resolution(const std::string &resolution);

/** Returns the Resolution enum variant from the given string. Throws an invalid_argument exception if resolution string is not valid. */
Resolution resolution_string_to_enum(const std::string &resolution);

/** Returns a string representation of the Resolution enum variant. */
std::string resolution_to_string(const Resolution &resolution);

/** Returns the height and width of the frames at the given resolution. */
std::tuple<int, int> get_height_and_width(const Resolution &resolution);

} // namespace rtsp
