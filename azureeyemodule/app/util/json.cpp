// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <string>

// Third party includees
#include "parson.h"

// Local includes
#include "json.hpp"

namespace json {

std::string object_to_string(const JSON_Object *obj)
{
    if (obj == nullptr)
    {
        util::log_error("Given a nullptr for JSON_Object in object_to_string function.");
        return "";
    }
    char *serialized_string = json_serialize_to_string_pretty((JSON_Value *)obj);
    std::string ret(serialized_string);
    json_free_serialized_string(serialized_string);
    return ret;
}

} // namespace json