/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This module acts as the interface between the azureeyemodule and the Azure IoT framework.
 */
#pragma once

#include <string>

namespace iot {
namespace msgs {

/** The telemetry channels we currently use. */
enum class MsgChannel
{
    NEURAL_NETWORK,
};

/**
 * Queue the given message to send to Azure IoT Hub.
 *
 * The given message should be valid JSON. We will then format the message as `{"<channel-as-str>": msg}`.
 *
 * @param channel: The message type we are sending. We will prepend a string representation of this
 *                 channel type to the message (while maintaining valid JSON formatting) so that messages
 *                 can be filtered downstream.
 * @param msg:     The actual message to send.
 */
void send_message(const MsgChannel &channel, const std::string &msg);

/** Set up the IoT thread. */
void start_iot();

/** Uninitialize the IoT library and join the thread. */
void stop_iot();

} // namespace msgs
} // namespace iot
