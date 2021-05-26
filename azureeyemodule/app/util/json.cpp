// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <string>

// Third party includees
#include "parson.h"

// Local includes
#include "json.hpp"

namespace json {

bool object_to_string(const JSON_Object *obj, std::string &ret)
{
    if (obj == nullptr)
    {
        util::log_error("Given a nullptr for JSON_Object in object_to_string function.");
        return false;
    }

    // Create a root value to assign this object to
    JSON_Value *root_value = json_value_init_object();

    char *serialized_string = json_serialize_to_string_pretty(root_value);
    if (serialized_string == nullptr)
    {
        util::log_error("Could not serialize an object into a string.");
        return false;
    }

    ret = std::string(serialized_string);
    json_free_serialized_string(serialized_string);
    json_value_free(root_value);
    return true;
}

} // namespace json