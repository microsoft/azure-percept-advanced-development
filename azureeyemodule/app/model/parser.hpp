// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

namespace model {
namespace parser {

/** Parser types we allow */
enum class Parser {
    CLASSIFICATION,
    DEFAULT,
    FASTER_RCNN_RESNET50,
    OBJECT_DETECTION,
    ONNXSSD,
    OPENPOSE,
    OCR,
    S1,
    SSD100,
    SSD200,
    UNET,
    YOLO
};

/**
 * Look up the parser based on the input argument
 *
 * @param parser_str The string representation of the parser.
 * @returns The enum representation of the parser.
 */
Parser from_string(const std::string &parser_str);

/**
 * Returns a string representation of the given parser.
 *
 * @param p The parser to convert.
 * @returns The string representation of the parser.
 */
std::string to_string(const Parser &p);

} // namespace parser
} // namespace model
