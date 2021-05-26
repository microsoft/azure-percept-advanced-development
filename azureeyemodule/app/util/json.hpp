// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#if (__cplusplus > 201703L) && (__GNUC__ > 9)
    // This header only exists in GCC > 9.x
    #include <concepts>
#endif
#include <string>

// Third party includes
#include "parson.h"

// Local includes
#include "helper.hpp"

namespace json {

#if (__cplusplus > 201703L) && (__GNUC__ > 9)
// Only if C++ version > 17 and if GCC > 9.x
template<typename T, typename ... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

template<typename T>
concept JsonType = IsAnyOf<std::remove_cvref_t<std::remove_pointer_t<std::decay_t<T>>>, bool, double, JSON_Array, JSON_Object, std::string, int, json_array_t, json_object_t>;
#elif (__cplusplus > 201703L) && (__GNUC__ == 9)
// Concepts are slightly different in GCC 9
template<typename T, typename ... U>
concept IsAnyOf = (std::is_same<T, U>::value || ...);

template<typename T>
concept JsonType = IsAnyOf<std::remove_cvref_t<std::remove_pointer_t<std::decay_t<T>>>, bool, double, JSON_Array, JSON_Object, std::string, int, json_array_t, json_object_t>;
#endif

/**
 * Checks if `key` is in `obj` and returns true if so, along with ret. Otherwise, ret is unfilled and returns false.
 *
 * Allowable types for T are:
 *
 * - bool
 * - double
 * - int
 * - JSON_Array pointer
 * - JSON_Object pointer
 * - std::string
 */
#if __cplusplus > 201703L
template<JsonType T>
#else
template<typename T>
#endif
bool _get_from_json_object(const JSON_Object *obj, const std::string &key, T &ret)
{
    if (json_object_dotget_value(obj, key.c_str()) != nullptr)
    {
        if (std::is_same<T, bool>::value)
        {
            int x = json_object_dotget_boolean(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, double>::value)
        {
            double x = json_object_dotget_number(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, JSON_Array*>::value)
        {
            JSON_Array *x = json_object_dotget_array(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, JSON_Object*>::value)
        {
            JSON_Object *x = json_object_dotget_object(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, std::string>::value)
        {
            const char *x = json_object_dotget_string(obj, key.c_str());
            std::string y(x);
            ret = reinterpret_cast<T&>(y);
        }
        else if (std::is_same<T, int>::value)
        {
            double x = json_object_dotget_number(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else
        {
            util::log_error("Given a JSON type that doesn't make sense.");
            return false;
        }
        return true;
    }
    else if (json_object_get_value(obj, key.c_str()) != nullptr)
    {
        if (std::is_same<T, bool>::value)
        {
            int x = json_object_get_boolean(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, double>::value)
        {
            double x = json_object_get_number(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, JSON_Array*>::value)
        {
            JSON_Array *x = json_object_get_array(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, JSON_Object*>::value)
        {
            JSON_Object *x = json_object_get_object(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else if (std::is_same<T, std::string>::value)
        {
            const char *x = json_object_get_string(obj, key.c_str());
            std::string y(x);
            ret = reinterpret_cast<T&>(y);
        }
        else if (std::is_same<T, int>::value)
        {
            double x = json_object_get_number(obj, key.c_str());
            ret = reinterpret_cast<T&>(x);
        }
        else
        {
            util::log_error("Given a JSON type that doesn't make sense.");
            return false;
        }
        return true;
    }
    else
    {
        return false;
    }
}

/**
 * Attempts to get the value associated with `key` from `json_as_str`, filling the result into `ret` if
 * successful and returning true. If it fails, `ret` is left alone and it returns false.
 *
 * `json_as_str` should be a JSON object - i.e., it should have curly braces surrounding it.
 * `key` can be an item in an object hierarchy using dot notation.
 *
 * Allowable types for T are:
 *
 * - bool
 * - double
 * - int
 * - JSON_Array pointer
 * - JSON_Object pointer
 * - std::string
 */
#if __cplusplus > 201703L
template<JsonType T>
#else
template<typename T>
#endif
bool try_parse_string(const std::string &json_as_str, const std::string &key, T &ret)
{
    JSON_Value *root_value = json_parse_string(json_as_str.c_str());
    if (root_value == nullptr)
    {
        util::log_error("Could not parse this string as JSON: " + json_as_str);
        return false;
    }

    JSON_Object *root_object = json_value_get_object(root_value);
    if (root_object == nullptr)
    {
        util::log_error("Could not parse out a top-level object from this JSON string: " + json_as_str);
        json_value_free(root_value);
        return false;
    }
    else
    {
        bool worked = _get_from_json_object<T>(root_object, key.c_str(), ret);
        json_value_free(root_value);
        return worked;
    }
}

/**
 * Attempts to get the value associated with `key` from `json_as_str`, returning `default_value` if it can't.
 *
 * `json_as_str` should be a JSON object - i.e., it should have curly braces surrounding it.
 * `key` can be an item in an object hierarchy using dot notation.
 *
 * Allowable types for T are:
 *
 * - bool
 * - double
 * - int
 * - JSON_Array pointer
 * - JSON_Object pointer
 * - std::string
 */
#if __cplusplus > 201703L
template<JsonType T>
#else
template<typename T>
#endif
T parse_string(const std::string &json_as_str, const std::string &key, T default_value)
{
    T ret;
    if (try_parse_string(json_as_str, key, ret))
    {
        return ret;
    }
    else
    {
        return default_value;
    }
}

/** Converts the given JSON_Object to a string. If we cannot serialize `obj` into a string, we print an error and return false, leaving `ret` unfilled. */
bool object_to_string(const JSON_Object *obj, std::string &ret);

} // namespace json