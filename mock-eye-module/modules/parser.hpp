// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

namespace parser {

/**
 * Parser types we allow.
 *
 * To add a new AI model, start by adding a new parser value here and then add
 * it to the switch statement in main.
 *
 * Note that you should add to the `look_up_parser()` function to make sure you
 * can read the parser type from the command line.
 */
enum class Parser {
    FASTER_RCNN,
    OPENPOSE,
    SSD100,
    SSD200,
    YOLO
};

/**
 * Look up the parser based on the input argument
 *
 * @param parser_str The string representation of the parser.
 * @returns The enum representation of the parser.
 */
Parser look_up_parser(const std::string &parser_str);

} // namespace parser
