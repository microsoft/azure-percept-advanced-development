// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <string>

// Third party includes
#include "iothub_module_client_ll.h"
#include "iothub_message.h"
#include "azure_c_shared_utility/threadapi.h"
#include "azure_c_shared_utility/crt_abstractions.h"
#include "azure_c_shared_utility/platform.h"
#include "azure_c_shared_utility/shared_util_options.h"
#include "iothub_client_options.h"
#include "iothubtransportmqtt.h"
#include "iothub.h"

// Local includes
#include "iot_update.hpp"
#include "../streaming/rtsp.hpp"
#include "../secure_ai/secureai.hpp"
#include "../util/helper.hpp"
#include "../util/json.hpp"

namespace iot {
namespace update {

/** The callback for updating the AI model. */
static update_cb_t update_cb = nullptr;

/** The callback for updating the image capture parameters. */
static update_collection_params_cb_t update_collection_params_cb = nullptr;

/** The callback for updating the telemetry intervals. */
static update_telemetry_interval_cb_t update_telemetry_interval_cb = nullptr;

/** The callback for updating the resolution in the model pipeline. */
static update_resolution_cb_t update_resolution_cb = nullptr;

/** This callback checks if the given model configuration JSON is different than the model's current one. */
static check_model_config_cb_t check_model_config_cb = nullptr;

/** Update the intervals for the telemetry channels. */
static void update_telemetry_intervals(unsigned long int nn_interval_ms)
{
    if (update_telemetry_interval_cb == nullptr)
    {
        util::log_info("Attempting to update telemetry intervals before a callback function is set.");
        return;
    }

    (*update_telemetry_interval_cb)(nn_interval_ms);
}

/** Update the retraining data collection parameters */
static void update_retraining_data_collection_parameters(bool enable, unsigned long int interval_seconds)
{
    if (update_collection_params_cb == nullptr)
    {
        util::log_info("Attempting to update the data collection parameters before a callback function is set.");
        return;
    }

    // Update the data collection parameters
    (*update_collection_params_cb)(enable, interval_seconds);
}

/** Update the neural network model using the callback */
static void update_model(const std::string &data, const std::string &model_config)
{
    if (update_cb == nullptr)
    {
        util::log_info("Attempting to update model before the model callback function is set.");
        return;
    }

    // Update the model
    (*update_cb)(data, false, model_config);
}

/** Update via the secure AI lifecycle pathway. */
static void update_model(const secure::SecureAIParams &params, const std::string &model_config)
{
    if (update_cb == nullptr)
    {
        util::log_info("Attempting to update the model with the secure AI functions before a callback function is set.");
        return;
    }

    // Update the model
    (*update_cb)(params.to_string(), true, model_config);
}

/** Checks if the model parameters are different from the ones we are currently using. */
static bool check_if_model_params_are_different(const std::string &model_config_json)
{
    if (check_model_config_cb == nullptr)
    {
        util::log_info("Attempting to check if model config is different before there is an associated callback function.");
        return false;
    }

    return (*check_model_config_cb)(model_config_json);
}

/** Parse out the logging stuff from the module twin. Here we update our log level based on module twin. */
static void parse_logging(const std::string &json_str)
{
    bool log_verbose;
    if (json::try_parse_string<bool>(json_str, "desired.Logging", log_verbose))
    {
        util::set_logging(log_verbose);
    }
}

/** Parse the retraining stuff out of the module twin. */
static void parse_retraining(const std::string &json_str)
{
    bool got_something = false;
    bool enable = false;
    int interval = 0;

    if (json::try_parse_string<bool>(json_str, "desired.RetrainingDataCollectionEnabled", enable))
    {
        got_something = true;
    }

    if (json::try_parse_string<int>(json_str, "desired.RetrainingDataCollectionInterval", interval))
    {
        got_something = true;
    }

    // Deal with signedness (the JSON value is an int, but we want a uint)
    const unsigned long int default_interval_seconds = 60;
    unsigned long int uinterval;
    if (interval < 0)
    {
        util::log_error("RetrainingDataCollectionInterval is negative, which does not make sense. Setting to a default value of " + std::to_string(default_interval_seconds));
        uinterval = default_interval_seconds;
    }
    else
    {
        uinterval = (unsigned long int)interval;
    }

    // Only update the retraining data collection parameters if we got something from the JSON.
    // Otherwise we let it keep going with whatever values it already had.
    if (got_something)
    {
        update_retraining_data_collection_parameters(enable, uinterval);
    }
}

/**
 * Parse out the AI model update stuff.
 *
 * If the user disabled the secure AI stuff, and we found any model parameter changed,
 * we attempt to update using the normal route.
 * If we are now enabled, and if any model parameters change,
 * we attempt to update using the secure AI route.
 */
static void parse_model_update(const std::string &json_str)
{
    JSON_Object *model_config = nullptr;
    std::string model_name = "";
    std::string model_version = "";
    std::string mm_server_url = "";
    std::string model_url = "";
    bool enable_secure_ai = false;
    bool download_model_from_mm_server = true;
    bool parameters_changed = false; // We don't want to update the model unless we changed something about its configuration

    // Get everything from the JSON
    json::try_parse_string<JSON_Object*>(json_str, "desired.ModelConfiguration", model_config);
    json::try_parse_string<std::string>(json_str, "desired.SCZ_MODEL_NAME", model_name);
    json::try_parse_string<std::string>(json_str, "desired.SCZ_MODEL_VERSION", model_version);
    json::try_parse_string<std::string>(json_str, "desired.SCZ_MM_SERVER_URL", mm_server_url);
    json::try_parse_string<bool>(json_str, "desired.SecureAILifecycleEnabled", enable_secure_ai);
    json::try_parse_string<bool>(json_str, "desired.DownloadSecuredModelFromMMServer", download_model_from_mm_server);
    json::try_parse_string<std::string>(json_str, "desired.ModelZipUrl", model_url);

    // Convert the model_config (if we have one) to a JSON string
    std::string model_config_json = "";
    if (model_config != nullptr)
    {
        bool converted = json::object_to_string(model_config, model_config_json);
        if (!converted)
        {
            util::log_error("Could not convert the model configuration from JSON object into a JSON string.");
        }
    }

    parameters_changed = secure::update_secure_model_params(mm_server_url, model_name, model_version, enable_secure_ai, download_model_from_mm_server, model_url);
    parameters_changed = parameters_changed || check_if_model_params_are_different(model_config_json);

    if (enable_secure_ai && parameters_changed)
    {
        if (!model_name.empty() && !model_version.empty() && !mm_server_url.empty())
        {
            if (download_model_from_mm_server || (!download_model_from_mm_server && !model_url.empty()))
            {
                // We have enabled secure AI and we find the model config changes from the previous version. So we should update using the secure route now.
                update_model(secure::get_model_params(), model_config_json);
            }
            else
            {
                util::log_error("Invalid secured AI model configuration. The value of \"ModelZipUrl\" should not be empty if download the model from external URL.");
            }
        }
        else
        {
            util::log_error("Invalid secured AI model configuration. The value of \"SCZ_MODEL_NAME\", \"SCZ_MODEL_VERSION\" and \"SCZ_MM_SERVER_URL\" should not be empty.");
        }
    }
    else if (!enable_secure_ai && parameters_changed)
    {
        // We have disabled secure AI and and we find the model config changes from the previous version. So we should update using the normal route now.
        update_model(model_url, model_config_json);
    }
}

/** Parse the RTSP stream stuff and set the RTSP stuff based on what we find in the module twin. */
static void parse_streams(const std::string &json_str)
{
    bool enable_raw_stream;
    if (json::try_parse_string<bool>(json_str, "desired.RawStream", enable_raw_stream))
    {
        rtsp::set_stream_params(rtsp::StreamType::RAW, enable_raw_stream);
    }

    bool enable_result_stream;
    if (json::try_parse_string<bool>(json_str, "desired.ResultStream", enable_result_stream))
    {
        rtsp::set_stream_params(rtsp::StreamType::RESULT, enable_result_stream);
    }

    int fps;
    if (json::try_parse_string<int>(json_str, "desired.StreamFPS", fps))
    {
        rtsp::set_stream_params(rtsp::StreamType::RAW, fps);
        rtsp::set_stream_params(rtsp::StreamType::RESULT, fps);
    }

    std::string resolution;
    std::string old_resolution = rtsp::get_resolution(rtsp::StreamType::RAW); // Right now all streams use the same resolution
    bool changed = false;
    if (json::try_parse_string<std::string>(json_str, "desired.StreamResolution", resolution))
    {
        if (rtsp::is_valid_resolution(resolution))
        {
            rtsp::set_stream_params(rtsp::StreamType::RAW, resolution);
            rtsp::set_stream_params(rtsp::StreamType::RESULT, resolution);
            changed = old_resolution != resolution;
        }
        else
        {
            util::log_error("Invalid resolution setting: " + resolution);
        }
    }

    // If the resolution changed, we need to restart the pipeline with the new resolution.
    if (changed)
    {
        util::log_info("Old resolution: \"" + old_resolution + "\", New resolution: \"" + resolution + "\". Values are different, so updating.");
        restart_model_with_new_resolution(resolution);
    }
    else
    {
        util::log_info("Old resolution: \"" + old_resolution + "\", New resolution: \"" + resolution + "\". Values are the same, so ignoring.");
    }
}

/** Parse telemetry stuff. */
static void parse_telemetry(const std::string &json_str)
{
    int nn_tel_interval_ms;
    if (json::try_parse_string<int>(json_str, "desired.TelemetryIntervalNeuralNetworkMs", nn_tel_interval_ms))
    {
        util::log_info("Telemetry interval for neural network messages (ms): " + std::to_string(nn_tel_interval_ms));
        update_telemetry_intervals(nn_tel_interval_ms);
    }
}

/** This is the callback for when the module twin changes. */
static void module_twin_callback(DEVICE_TWIN_UPDATE_STATE update_state, const unsigned char *payload, size_t size, void *user_context_cb)
{
    util::log_info("module twin callback called with (state=" + std::string(MU_ENUM_TO_STRING(DEVICE_TWIN_UPDATE_STATE, update_state)) + ", size=" + std::to_string(size) + "): " + reinterpret_cast<const char*>(payload));

    std::string json_str(reinterpret_cast<const char *>(payload));

    // Parse out all the stuff and deal with it
    parse_logging(json_str);
    parse_retraining(json_str);
    parse_telemetry(json_str);
    parse_model_update(json_str);
    parse_streams(json_str);
}

void restart_model_with_new_resolution(const std::string &resolution)
{
    if (update_resolution_cb == nullptr)
    {
        util::log_info("Attempting to update the resolution before the model is running.");
        return;
    }

    // Update
    (*update_resolution_cb)(resolution);
}

void initialize(IOTHUB_MODULE_CLIENT_LL_HANDLE client_handle)
{
    auto ret = IoTHubModuleClient_LL_SetModuleTwinCallback(client_handle, module_twin_callback, (void*)client_handle);
    if (ret != IOTHUB_CLIENT_OK)
    {
        util::log_error("Could not initialize the module twin callback.");
        return;
    }
}

void set_update_callback(update_cb_t callback)
{
    update_cb = callback;
}

void set_update_collection_params_callback(update_collection_params_cb_t callback)
{
    update_collection_params_cb = callback;
}

void set_update_resolution_callback(update_resolution_cb_t callback)
{
    update_resolution_cb = callback;
}

void set_update_telemetry_intervals_callback(update_telemetry_interval_cb_t callback)
{
    update_telemetry_interval_cb = callback;
}

void set_check_model_config_callback(check_model_config_cb_t callback)
{
    check_model_config_cb = callback;
}

} // namespace update
} // namespace iot
