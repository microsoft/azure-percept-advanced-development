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
    if (parser_str == "ssd")
    {
        return Parser::SSD;
    }
    else
    {
        std::cerr << "Given " << parser_str << " for --parser, but we do not support it." << std::endl;
        exit(-1);
    }
}

} // namespace parser
