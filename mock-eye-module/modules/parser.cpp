// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <iostream>
#include <string>

// Third party includes
#include "parser.hpp"

namespace parser {

Parser look_up_parser(const std::string &parser_str)
{
    if (parser_str == "openpose")
    {
        return Parser::OPENPOSE;
    }
    else if (parser_str == "ssd100")
    {
        return Parser::SSD100;
    }
    else if (parser_str == "ssd200")
    {
        return Parser::SSD200;
    }
    else if (parser_str == "yolo")
    {
        return Parser::YOLO;
    }
    else if (parser_str == "faster-rcnn")
    {
        return Parser::FASTER_RCNN;
    }
    else
    {
        std::cerr << "Given " << parser_str << " for --parser, but we do not support it." << std::endl;
        exit(-1);
    }
}

} // namespace parser
