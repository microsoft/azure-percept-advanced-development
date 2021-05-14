// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <iostream>
#include <string>

#include "parser.hpp"

namespace model {
namespace parser {

Parser from_string(const std::string &parser_str)
{
    if (parser_str == "openpose")
    {
        return Parser::OPENPOSE;
    }
    else if (parser_str == "ocr")
    {
        return Parser::OCR;
    }
    else if (parser_str == "classification")
    {
        return Parser::CLASSIFICATION;
    }
    else if (parser_str == "objectdetection")
    {
        return Parser::OBJECT_DETECTION;
    }
    else if (parser_str == "default")
    {
        return Parser::DEFAULT;
    }
    else if (parser_str == "s1")
    {
        return Parser::S1;
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
    else if (parser_str == "onnxssd")
    {
        return Parser::ONNXSSD;
    }
    else if (parser_str == "unet")
    {
        return Parser::UNET;
    }
    else if (parser_str == "faster-rcnn-resnet50")
    {
        return Parser::FASTER_RCNN_RESNET50;
    }
    else
    {
        std::cerr << "Given " << parser_str << " for --parser, but we do not support it." << std::endl;
        exit(__LINE__);
    }
}

std::string to_string(const Parser &p)
{
    switch (p)
    {
        case Parser::CLASSIFICATION:
            return "classification";
        case Parser::DEFAULT:
            return "default";
        case Parser::OBJECT_DETECTION:
            return "objectdetection";
        case Parser::OPENPOSE:
            return "openpose";
        case Parser::OCR:
            return "ocr";
        case Parser::S1:
            return "s1";
        case Parser::SSD100:
            return "ssd100";
        case Parser::SSD200:
            return "ssd200";
        case Parser::YOLO:
            return "yolo";
        case Parser::ONNXSSD:
            return "onnxssd";
        case Parser::UNET:
            return "unet";
        case Parser::FASTER_RCNN_RESNET50:
            return "faster-rcnn-resnet50";
        default:
            std::cerr << "Can't convert this type of parser to a string." << std::endl;
            exit(__LINE__);
    }
}

} // namespace parser
} // namespace model
