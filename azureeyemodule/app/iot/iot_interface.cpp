// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>

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
#include "iot_interface.hpp"
#include "iot_update.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/circular_buffer.hpp"
#include "../util/helper.hpp"
#include "../util/timing.hpp"

namespace iot {
namespace msgs {

/**
 * A C-style struct to be used by the low level SDK to send and track the messages.
 */
typedef struct
{
    /** Handle for the message. */
    IOTHUB_MESSAGE_HANDLE handle;

    /** In case we want to track exactly which message in the user-defined send-message callback. */
    size_t tracking_id;
} ll_msg_t;

/** The types of messages we can send to the IoT loop thread. */
enum class IotMsgType
{
    STOP_IOT,
    SEND_MSG,
};

/** These are the messages that we send to the IoT loop thread. */
struct IotMsg
{
    /** Totally default constructor for those times you need one. */
    IotMsg() : type(IotMsgType::STOP_IOT), msg_channel(MsgChannel::NEURAL_NETWORK), msg_body("") {};

    /** Constructor for non- send-message -type messages. We fill in the stuff that's specific to send-message type of messages. */
    IotMsg(const IotMsgType &type) : type(type), msg_channel(MsgChannel::NEURAL_NETWORK), msg_body("") {};

    /** Constructor for the send-message type of message. */
    IotMsg(const IotMsgType &type, const MsgChannel &channel, const std::string &msg_body) : type(type), msg_channel(channel), msg_body(msg_body) {};

    IotMsgType type;
    MsgChannel msg_channel;
    std::string msg_body;
};

/** The maximum messages we can send to the IoT thread before we start overwriting old ones. */
static const size_t max_thread_msgs = 1000;

/** The mailbox for the IoT thread. We should definitely be able to remove these as fast as they come in. */
static circbuf::CircularBuffer<IotMsg> mailbox(max_thread_msgs);

/** The IoT thread. */
static std::thread iotthread;

/** This is used to signal to the stop_iot() function that we have actually stopped and therefore can be joined. */
static std::condition_variable stop_condition;

/** This gets switched to true when we are exiting the IoT thread. Ugh, C++ has so much boilerplate... */
static bool thread_ready_to_join = false;

/** Mapping of all the message channel types to the interval in ms for that message type. */
static std::unordered_map<MsgChannel, unsigned long int> telemetry_intervals_ms = {
    { MsgChannel::NEURAL_NETWORK, 1000 }
};

static std::unordered_map<MsgChannel, ourtime::Timer> telemetry_timers = {
    { MsgChannel::NEURAL_NETWORK, ourtime::Timer() }
};

/** This callback gets called whenever we send a message (or try to) to the Azure IoT Hub. */
static void azure_iot_send_msg_callback(IOTHUB_CLIENT_CONFIRMATION_RESULT result, void *user_context_cb)
{
}

/**
 * This is the function that gets called upon receiving a value for telemetry intervals
 * in the module twin update.
 *
 * @param nn_msg_interval_ms: We only send messages from the neural network every this many ms. All
 *                            other messages from the neural network are discarded.
 */
static void update_telemetry_values(unsigned long int nn_msg_interval_ms)
{
    telemetry_intervals_ms[MsgChannel::NEURAL_NETWORK] = nn_msg_interval_ms;
}

/** STOP_IOT message handler. Uninitializes the Azure IoT SDK. */
static void stop_iot_loop(IOTHUB_MODULE_CLIENT_LL_HANDLE client_handle)
{
    if (client_handle != nullptr)
    {
        IoTHubModuleClient_LL_Destroy(client_handle);
        IoTHub_Deinit();
    }

    // Signal the waiting stop() function that it can join() the thread now.
    thread_ready_to_join = true;
    stop_condition.notify_one();
}

/** Return true if we should send this type of message (i.e., if this type of message's timer has expired). */
static bool filter_msg_by_telemetry_intervals(const MsgChannel &channel)
{
    if (telemetry_timers[channel].elapsed_ms() > telemetry_intervals_ms[channel])
    {
        telemetry_timers[channel].reset();
        return true;
    }
    else
    {
        return false;
    }
}

/** RECV_MSG message handler. Handles incoming messages for the ONVIF channel. */
static IOTHUBMESSAGE_DISPOSITION_RESULT receive_onvif_message(IOTHUB_MESSAGE_HANDLE message, void *unused)
{
    const unsigned char* buffer = nullptr;
    size_t s;
    if (IoTHubMessage_GetByteArray(message, &buffer, &s) != IOTHUB_MESSAGE_OK)
    {
        util::log_error("Try to get  message content but failed.");
        return IOTHUBMESSAGE_REJECTED;
    }
    char* str = (char *) malloc(s + 1);
    memcpy(str, buffer, s);
    str[s] = '\0';
    JSON_Value *root_value = json_parse_string(reinterpret_cast<const char*>(str));
    JSON_Object *root_object = json_value_get_object(root_value);
    util::log_info("Got an ONVIF control message: " + std::string(str));
    free(str);

    // Set stuff based on results
    if (json_object_get_value(root_object, "snapshot") != nullptr)
    {
        std::string type  = json_object_get_string(root_object, "snapshot");
        if (type == "raw")
        {
            rtsp::take_snapshot(rtsp::StreamType::RAW);
        }
        else if (type == "result")
        {
            rtsp::take_snapshot(rtsp::StreamType::RESULT);
        }
        else
        {
            util::log_error("Receive a invalid snapshot type request: " + type);
            return IOTHUBMESSAGE_REJECTED;
        }
    }


    if (json_object_get_value(root_object, "-fps") != nullptr)
    {
        std::string value  = json_object_get_string(root_object, "-fps");
        try
        {
            rtsp::set_stream_params(rtsp::StreamType::RAW, std::stoi(value));
            rtsp::set_stream_params(rtsp::StreamType::RESULT, std::stoi(value));
        }
        catch (std::invalid_argument &e)
        {
            // catch the error if the fps parameter can't be converted to int
            util::log_error("While parsing ONVIF message, could not convert FPS parameter into an integer: " + value);
            return IOTHUBMESSAGE_REJECTED;
        }
    }

    if (json_object_get_value(root_object, "-s") != nullptr)
    {
        std::string value = json_object_get_string(root_object, "-s");
        if (rtsp::is_valid_resolution(std::string(value)))
        {
            // change the model resolution
            iot::update::restart_model_with_new_resolution(rtsp::resolution_string_to_enum(std::string(value)));

            // reset the rtsp stream resolution, and then disconnect the stream to let the new parameter can be loaded
            rtsp::set_stream_params(rtsp::StreamType::RAW, rtsp::resolution_string_to_enum(std::string(value)), false);
            rtsp::set_stream_params(rtsp::StreamType::RESULT, rtsp::resolution_string_to_enum(std::string(value)), false);

            // restart the rtsp stream
            rtsp::set_stream_params(rtsp::StreamType::RAW,  true);
            rtsp::set_stream_params(rtsp::StreamType::RESULT, true);
        }
        else
        {
            util::log_error("While parsing ONVIF message, found invalid resolution value of " + value);
            return IOTHUBMESSAGE_REJECTED;
        }
    }

    return IOTHUBMESSAGE_ACCEPTED;
}

/** SEND_MSG message handler. Sends the given message over the Azure IoT connection. */
static void send_iot_msg(IOTHUB_MODULE_CLIENT_LL_HANDLE client_handle, const MsgChannel &channel, std::string msg)
{
    // Filter out messages unless their telemetry interval timer has expired
    bool should_send = filter_msg_by_telemetry_intervals(channel);
    if (!should_send)
    {
        // We discard this message because the user only wants us to send this type
        // of message every N ms, and the number of ms has not transpired yet.
        return;
    }

    ll_msg_t message_instance;
    message_instance.handle = IoTHubMessage_CreateFromString(msg.c_str());
    if (message_instance.handle == nullptr)
    {
        util::log_error("Could not create an IoT message to send.");
        return;
    }

    static size_t message_tracking_id = 0;
    message_instance.tracking_id = message_tracking_id++; // Overflow is fine, we'll just wrap since we are unsigned.

    IOTHUB_MESSAGE_RESULT result;
    result = IoTHubMessage_SetMessageId(message_instance.handle, "MSG_ID");
    if (result != IOTHUB_MESSAGE_OK)
    {
        util::log_error("Could not set message ID for Azure IoT message we are trying to send. Msg ID: " + std::to_string(message_instance.tracking_id));
    }

    result = IoTHubMessage_SetCorrelationId(message_instance.handle, "CORE_ID");
    if (result != IOTHUB_MESSAGE_OK)
    {
        util::log_error("Could not set correlation ID for Azure IoT message we are trying to send. Msg ID: " + std::to_string(message_instance.tracking_id));
    }

    result = IoTHubMessage_SetContentTypeSystemProperty(message_instance.handle, "application%2fjson");
    if (result != IOTHUB_MESSAGE_OK)
    {
        util::log_error("Could not set content type for Azure IoT message we are trying to send. Msg ID: " + std::to_string(message_instance.tracking_id));
    }

    result = IoTHubMessage_SetContentEncodingSystemProperty(message_instance.handle, "utf-8");
    if (result != IOTHUB_MESSAGE_OK)
    {
        util::log_error("Could not set content encoding for Azure IoT message we are trying to send. Msg ID: " + std::to_string(message_instance.tracking_id));
    }

    auto res = IoTHubModuleClient_LL_SendEventToOutputAsync(client_handle, message_instance.handle, "AzureEyeModuleOutput", azure_iot_send_msg_callback, &message_instance);
    if (res != IOTHUB_CLIENT_OK)
    {
        util::log_error("Could not send an Azure IoT message. Msg ID: " + std::to_string(message_instance.tracking_id));
    }
}

/**
 * This function runs as a forever loop, accepting messages from the API and acting on them.
 * The reason for this is that we need to call the do_work function at least once every 100ish ms,
 * and if we call it as part of our main loop, we may not meet timing, especially since the main
 * loop deals with neural network inference, which may take seconds at a time.
 */
static void iot_loop(IOTHUB_MODULE_CLIENT_LL_HANDLE client_handle)
{
    while (true)
    {
        // Pull the next message from the mailbox queue
        IotMsg msg;
        const auto timeout_ms = 50;
        bool got_a_msg = mailbox.get_with_timeout(msg, timeout_ms);
        if (got_a_msg)
        {
            // Act on the message we just got
            switch (msg.type)
            {
                case IotMsgType::STOP_IOT:
                    stop_iot_loop(client_handle);
                    // Now return to let this thread join
                    return;
                case IotMsgType::SEND_MSG:
                    send_iot_msg(client_handle, std::move(msg.msg_channel), std::move(msg.msg_body));
                    break;
                default:
                    util::log_error("IoT loop got a type of message it does not understand.");
                    break;
            }
        }

        // The whole point of making this thread is that have to service the SDK
        // via this function every ~100ish ms.
        IoTHubModuleClient_LL_DoWork(client_handle);
    }
}

static std::string channel_to_string(const MsgChannel &ch)
{
    switch (ch)
    {
        case MsgChannel::NEURAL_NETWORK:
            return "NEURAL_NETWORK";
        default:
            util::log_error("Need to implement a string representation of this message channel.");
            return "";
    }
}

void start_iot()
{
    // Try to start the IoT library
    auto ret = IoTHub_Init();
    if (ret != 0)
    {
        util::log_error("Failed to initialize the Azure IoT SDK.");
        return;
    }

    // Get a client handle that we need for all the IoT functions
    IOTHUB_MODULE_CLIENT_LL_HANDLE client_handle = IoTHubModuleClient_LL_CreateFromEnvironment(MQTT_Protocol);
    if (client_handle == nullptr)
    {
        util::log_error("Could not create an Azure IoT SDK client handle from environment.");
        return;
    }

    // Set up our module twin callback
    update::initialize(client_handle);

    // Set up our telemetry intervals callback
    update::set_update_telemetry_intervals_callback(update_telemetry_values);

    // Set up the ONVIF module callback
    if (IoTHubModuleClient_LL_SetInputMessageCallback(client_handle, "onvif-control-msg-input", receive_onvif_message, nullptr) != IOTHUB_CLIENT_OK)
    {
        util::log_error("Could not create a callback to handle the ONVIF module control messages.");
    }

    // Start the IoT thread
    iotthread = std::thread(iot_loop, client_handle);
}

void stop_iot()
{
    // A silly mutex to make the stop_condition API happy.
    // Since we only read thread_ready_to_join, and since the IoT thread only
    // writes to that variable, we shouldn't really need a mutex, as there is no
    // chance for a race condition.
    static std::mutex condition_mutex;

    if (iotthread.joinable())
    {
        // Since we are communicating with the IoT thread using a circular buffer
        // with a fixed capacity, it is possible that we could send the kill signal to it
        // and that it doesn't get around to servicing the kill signal before the signal gets
        // overwritten by other messages coming in. So, we need to send the kill message,
        // then wait until we are signaled that the thread is ready to join(). We do this
        // with a timeout, and keep trying until it works or we try so many times that
        // it is clear it will not work in a reasonable amount of time.
        const auto max_kill_tries = 10; // arbitrarily chosen - the best way to choose stuff!
        for (auto i = 0; i < max_kill_tries; i++)
        {
            // Send the kill message
            util::log_info("Killing IoT connection.");
            IotMsg msg(IotMsgType::STOP_IOT);
            mailbox.put(std::move(msg));

            // Grab the condition mutex and wait for the condition.
            std::unique_lock<std::mutex> lock(condition_mutex);
            stop_condition.wait_for(lock, std::chrono::milliseconds(100), []{return thread_ready_to_join;});
            lock.unlock();

            if (thread_ready_to_join)
            {
                break;
            }
        }

        if (thread_ready_to_join)
        {
            // Wait for thread to join
            iotthread.join();
            thread_ready_to_join = false;
        }
        else
        {
            util::log_error("Could not kill the IoT thread. It is likely that it is being spammed with messages or that the communication buffer with it is too small.");
        }
    }
    else
    {
        util::log_info("Trying to kill IoT connection, but we are not currently connected.");
    }
}

void send_message(const MsgChannel &channel, const std::string &msg)
{
    IotMsg iotmsg(IotMsgType::SEND_MSG, channel, "{\"" + channel_to_string(channel) + "\": " + msg + "}");
    mailbox.put(std::move(iotmsg));
}

} // namespace msgs
} // namespace iot
