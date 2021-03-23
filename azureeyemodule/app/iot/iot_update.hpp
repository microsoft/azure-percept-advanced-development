/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This module is responsible for the module twin update mechanism.
 */
#pragma once

// Local includes
#include "../streaming/resolution.hpp"

// Standard library includes
#include <string>

// Third party includes
#include "iothub_module_client_ll.h"

namespace iot {
namespace update {

/** A convenience typedef for the type of function we need as an AI model update function. */
typedef void (*update_cb_t)(const std::string &, bool);

/** A convenience typedef for the type of function we need as a data collection parameter change callback. */
typedef void (*update_collection_params_cb_t)(bool, unsigned long int);

/** A convenience typedef for the type of function we need as a telemetry update callback. First arg is the neural network telemetry interval in ms. */
typedef void (*update_telemetry_interval_cb_t)(unsigned long int);

/** A convenience typedef for the type of function we need to update the resolution in the AI model. */
typedef void (*update_resolution_cb_t)(const rtsp::Resolution &);

/** A convenience typedef for the type of function we need to update the time alignment feature. */
typedef void (*update_time_alignment_cb_t)(bool);

/** Initialize the module twin update callback using the given IoT handle. */
void initialize(IOTHUB_MODULE_CLIENT_LL_HANDLE client_handle);

/** Restart the AI model with the new resolution. */
void restart_model_with_new_resolution(const rtsp::Resolution &resolution);

/** Set the update callback function. This callback will be called to alert us to update the AI model. */
void set_update_callback(update_cb_t callback);

/** Set the callback that happens when the retraining data collection parameters change as a result of module twin update. */
void set_update_collection_params_callback(update_collection_params_cb_t callback);

/** Set the callback that happens when the resolution changes. */
void set_update_resolution_callback(update_resolution_cb_t callback);

/** Set the callback function that gets called when the telemetry intervals update. */
void set_update_telemetry_intervals_callback(update_telemetry_interval_cb_t callback);

/** Set the callback function that gets called when the time alignment feature gets updated. */
void set_update_time_alignment_callback(update_time_alignment_cb_t callback);

} // namespace update
} // namespace iot
