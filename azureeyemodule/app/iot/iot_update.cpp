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
#include "parson.h"

// Local includes
#include "iot_update.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../secure_ai/secureai.hpp"

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

/** The callback function for updating the time alignment feature. */
static update_time_alignment_cb_t update_time_alignment_cb = nullptr;

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
static void update_model(const std::string &data)
{
    if (update_cb == nullptr)
    {
        util::log_info("Attempting to update model before the model callback function is set.");
        return;
    }

    // Update the model
    (*update_cb)(data, false);
}

/** Update via the secure AI lifecycle pathway. */
static void update_model(const secure::SecureAIParams &params)
{
    if (update_cb == nullptr)
    {
        util::log_info("Attempting to update the model with the secure AI functions before a callback function is set.");
        return;
    }

    // Update the model
    (*update_cb)(params.to_string(), true);
}

/** Update the time alignment feature based on the callback we are given from main. */
static void update_time_alignment(bool align)
{
    if (update_time_alignment_cb == nullptr)
    {
        util::log_info("Attempting to update the time alignment feature before a callback function is set.");
        return;
    }

    // Update
    (*update_time_alignment_cb)(align);
}

/** Parse out the logging stuff from the module twin. Here we update our log level based on module twin. */
static void parse_logging(JSON_Object *root_object)
{
    if (json_object_dotget_value(root_object, "desired.Logging") != nullptr)
    {
        util::set_logging(json_object_dotget_boolean(root_object, "desired.Logging"));
    }
    if (json_object_get_value(root_object, "Logging") != nullptr)
    {
        util::set_logging(json_object_get_boolean(root_object, "Logging"));
    }
}

/** Parse the retraining stuff out of the module twin. */
static void parse_retraining(JSON_Object *root_object)
{
    bool got_something = false;
    bool enable = false;
    int interval = 0;

    if (json_object_dotget_value(root_object, "desired.RetrainingDataCollectionEnabled") != nullptr)
    {
        enable = json_object_dotget_boolean(root_object, "desired.RetrainingDataCollectionEnabled");
        got_something = true;
    }
    if (json_object_get_value(root_object, "RetrainingDataCollectionEnabled") != nullptr)
    {
        enable = json_object_get_boolean(root_object, "RetrainingDataCollectionEnabled");
        got_something = true;
    }

    if (json_object_dotget_value(root_object, "desired.RetrainingDataCollectionInterval") != nullptr)
    {
        interval = json_object_dotget_number(root_object, "desired.RetrainingDataCollectionInterval");
        got_something = true;
    }
    if (json_object_get_value(root_object, "RetrainingDataCollectionInterval") != nullptr)
    {
        interval = json_object_get_number(root_object, "RetrainingDataCollectionInterval");
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
static void parse_model_update(JSON_Object *root_object)
{
    std::string model_name = "";
    std::string model_version = "";
    std::string mm_server_url = "";
    std::string model_url = "";
    bool enable_secure_ai = false;
    bool download_model_from_mm_server = true;
    bool parameters_changed = false; // We don't want to update the model unless we changed something about its configuration

    if (json_object_dotget_value(root_object, "desired.SCZ_MODEL_NAME") != nullptr)
    {
        model_name = json_object_dotget_string(root_object, "desired.SCZ_MODEL_NAME");
    }
    if (json_object_get_value(root_object, "SCZ_MODEL_NAME") != nullptr)
    {
        model_name = json_object_get_string(root_object, "SCZ_MODEL_NAME");
    }

    if (json_object_dotget_value(root_object, "desired.SCZ_MODEL_VERSION") != nullptr)
    {
        model_version = json_object_dotget_string(root_object, "desired.SCZ_MODEL_VERSION");
    }
    if (json_object_get_value(root_object, "SCZ_MODEL_VERSION") != nullptr)
    {
        model_version = json_object_get_string(root_object, "SCZ_MODEL_VERSION");
    }

    if (json_object_dotget_value(root_object, "desired.SCZ_MM_SERVER_URL") != nullptr)
    {
        mm_server_url = json_object_dotget_string(root_object, "desired.SCZ_MM_SERVER_URL");
    }
    if (json_object_get_value(root_object, "SCZ_MM_SERVER_URL") != nullptr)
    {
        mm_server_url = json_object_get_string(root_object, "SCZ_MM_SERVER_URL");
    }

    if (json_object_dotget_value(root_object, "desired.SecureAILifecycleEnabled") != nullptr)
    {
        enable_secure_ai = json_object_dotget_boolean(root_object, "desired.SecureAILifecycleEnabled");
    }
    if (json_object_get_value(root_object, "SecureAILifecycleEnabled") != nullptr)
    {
        enable_secure_ai = json_object_get_boolean(root_object, "SecureAILifecycleEnabled");
    }

    if (json_object_dotget_value(root_object, "desired.DownloadSecuredModelFromMMServer") != nullptr)
    {
        download_model_from_mm_server = json_object_dotget_boolean(root_object, "desired.DownloadSecuredModelFromMMServer");
    }
    if (json_object_get_value(root_object, "DownloadSecuredModelFromMMServer") != nullptr)
    {
        download_model_from_mm_server = json_object_get_boolean(root_object, "DownloadSecuredModelFromMMServer");
    }

    if (json_object_dotget_value(root_object, "desired.ModelZipUrl") != nullptr)
    {
        model_url = json_object_dotget_string(root_object, "desired.ModelZipUrl");
    }
    if (json_object_get_value(root_object, "ModelZipUrl") != nullptr)
    {
        model_url = json_object_get_string(root_object, "ModelZipUrl");
    }

    // BLOCKS until it can get the mutex for the secure AI params.
    util::log_debug("Accessing update_secure_model_params mutex from IoT update.");
    parameters_changed = secure::update_secure_model_params(mm_server_url, model_name, model_version, enable_secure_ai, download_model_from_mm_server, model_url);
    util::log_debug("Done accessing update_secure_model_params mutex from IoT update.");

    if (enable_secure_ai && parameters_changed)
    {
        if (!model_name.empty() && !model_version.empty() && !mm_server_url.empty())
        {
            if (download_model_from_mm_server || (!download_model_from_mm_server && !model_url.empty()))
            {
                // We have enabled secure AI and we find the model config changes from the previous version. So we should update using the secure route now.
                // BLOCKS until it can get the mutex for the secure AI params.
                util::log_debug("Accessing get_model_params mutex from IoT update.");
                update_model(secure::get_model_params());
                util::log_debug("Done accessing get_model_params mutex from IoT Update.");
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
        update_model(model_url);
    }
}

/** Parse the RTSP stream stuff and set the RTSP stuff based on what we find in the module twin. */
static void parse_streams(JSON_Object *root_object)
{
    if (json_object_dotget_value(root_object, "desired.RawStream") != nullptr)
    {
        rtsp::set_stream_params(rtsp::StreamType::RAW, (bool)json_object_dotget_boolean(root_object, "desired.RawStream"));
    }
    if (json_object_get_value(root_object, "RawStream") != nullptr)
    {
        rtsp::set_stream_params(rtsp::StreamType::RAW, (bool)json_object_dotget_boolean(root_object, "RawStream"));
    }

    if (json_object_dotget_value(root_object, "desired.ResultStream") != nullptr)
    {
        rtsp::set_stream_params(rtsp::StreamType::RESULT, (bool)json_object_dotget_boolean(root_object, "desired.ResultStream"));
    }
    if (json_object_get_value(root_object, "ResultStream") != nullptr)
    {
        rtsp::set_stream_params(rtsp::StreamType::RESULT, (bool)json_object_dotget_boolean(root_object, "ResultStream"));
    }

    if (json_object_dotget_value(root_object, "desired.H264Stream") != nullptr)
    {
        rtsp::set_stream_params(rtsp::StreamType::H264_RAW, (bool)json_object_dotget_boolean(root_object, "desired.H264Stream"));
    }
    if (json_object_dotget_value(root_object, "H264Stream") != nullptr)
    {
        rtsp::set_stream_params(rtsp::StreamType::H264_RAW, (bool)json_object_dotget_boolean(root_object, "H264Stream"));
    }

    if (json_object_dotget_value(root_object, "desired.StreamFPS") != nullptr)
    {
        auto fps = json_object_dotget_number(root_object, "desired.StreamFPS");
        rtsp::set_stream_params(rtsp::StreamType::RAW, (int)fps);
        rtsp::set_stream_params(rtsp::StreamType::RESULT, (int)fps);
    }
    if (json_object_dotget_value(root_object, "StreamFPS") != nullptr)
    {
        auto fps = json_object_dotget_number(root_object, "StreamFPS");
        rtsp::set_stream_params(rtsp::StreamType::RAW, (int)fps);
        rtsp::set_stream_params(rtsp::StreamType::RESULT, (int)fps);
    }

    std::string resolution;
    rtsp::Resolution old_resolution = rtsp::get_resolution(rtsp::StreamType::RAW); // Right now all streams use the same resolution
    bool changed = false;
    if (json_object_dotget_value(root_object, "desired.StreamResolution") != nullptr)
    {
        resolution = std::string(json_object_dotget_string(root_object, "desired.StreamResolution"));
        if (rtsp::is_valid_resolution(std::string(resolution)))
        {
            rtsp::Resolution new_resolution = rtsp::resolution_string_to_enum(resolution);
            rtsp::set_stream_params(rtsp::StreamType::RAW, new_resolution);
            rtsp::set_stream_params(rtsp::StreamType::RESULT, new_resolution);
            changed = old_resolution != new_resolution;
        }
        else
        {
            util::log_error("Invalid resolution setting: " + std::string(resolution));
        }
    }
    if (json_object_dotget_value(root_object, "StreamResolution") != nullptr)
    {
        resolution = std::string(json_object_dotget_string(root_object, "StreamResolution"));
        if (rtsp::is_valid_resolution(std::string(resolution)))
        {
            rtsp::Resolution new_resolution = rtsp::resolution_string_to_enum(resolution);
            rtsp::set_stream_params(rtsp::StreamType::RAW, new_resolution);
            rtsp::set_stream_params(rtsp::StreamType::RESULT, new_resolution);
            changed = old_resolution != new_resolution;
        }
        else
        {
            util::log_error("Invalid resolution setting: " + std::string(resolution));
        }
    }

    // If the resolution changed, we need to restart the pipeline with the new resolution.
    if (changed)
    {
        util::log_info("Old resolution: \"" + rtsp::resolution_to_string(old_resolution) + "\", New resolution: \"" + resolution + "\". Values are different, so updating.");
        restart_model_with_new_resolution(rtsp::resolution_string_to_enum(resolution));
    }
    else
    {
        util::log_info("Old resolution: \"" + rtsp::resolution_to_string(old_resolution) + "\", New resolution: \"" + resolution + "\". Values are the same, so ignoring.");
    }
}

/** Parse telemetry stuff. */
static void parse_telemetry(JSON_Object *root_object)
{
    if (json_object_dotget_value(root_object, "desired.TelemetryIntervalNeuralNetworkMs") != nullptr)
    {
        auto nn_tel_interval_ms = json_object_dotget_number(root_object, "desired.TelemetryIntervalNeuralNetworkMs");
        util::log_info("Telemetry interval for neural network messages (ms): " + std::to_string(nn_tel_interval_ms));
        update_telemetry_intervals(nn_tel_interval_ms);
    }
    if (json_object_get_value(root_object, "TelemetryIntervalNeuralNetworkMs") != nullptr)
    {
        auto nn_tel_interval_ms = json_object_get_number(root_object, "TelemetryIntervalNeuralNetworkMs");
        util::log_info("Telemetry interval for neural network messages (ms): " + std::to_string(nn_tel_interval_ms));
        update_telemetry_intervals(nn_tel_interval_ms);
    }
}

/** Parse out and deal with the RTSP time alignment feature. */
static void parse_time_alignment(JSON_Object *root_object)
{
    if (json_object_dotget_value(root_object, "desired.TimeAlignRTSP") != nullptr)
    {
        bool time_align = json_object_dotget_boolean(root_object, "desired.TimeAlignRTSP");
        util::log_info("Time Align RTSP Streams: " + (time_align ? std::string("yes") : std::string("no")));
        update_time_alignment(time_align);
    }
    if (json_object_get_value(root_object, "TimeAlignRTSP") != nullptr)
    {
        bool time_align = json_object_dotget_boolean(root_object, "TimeAlignRTSP");
        util::log_info("Time Align RTSP Streams: " + (time_align ? std::string("yes") : std::string("no")));
        update_time_alignment(time_align);
    }
}

/** This is the callback for when the module twin changes. */
static void module_twin_callback(DEVICE_TWIN_UPDATE_STATE update_state, const unsigned char *payload, size_t size, void *user_context_cb)
{
    util::log_info("module twin callback called with (state=" + std::string(MU_ENUM_TO_STRING(DEVICE_TWIN_UPDATE_STATE, update_state)) + ", size=" + std::to_string(size) + "): " + reinterpret_cast<const char*>(payload));

    JSON_Value *root_value = json_parse_string(reinterpret_cast<const char*>(payload));
    JSON_Object *root_object = json_value_get_object(root_value);

    // Parse out all the stuff and deal with it
    parse_logging(root_object);
    parse_retraining(root_object);
    parse_telemetry(root_object);
    parse_model_update(root_object);
    parse_streams(root_object);
    parse_time_alignment(root_object);
}

void restart_model_with_new_resolution(const rtsp::Resolution &resolution)
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

void set_update_time_alignment_callback(update_time_alignment_cb_t callback)
{
    update_time_alignment_cb = callback;
}

} // namespace update
} // namespace iot
