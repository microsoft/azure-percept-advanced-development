// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <thread>

// Third party library includes
#include <libusb-1.0/libusb.h>

// Local includes
#include "device.hpp"
#include "validator.h"
#include "../util/helper.hpp"


namespace device {

// STM32
static const int mcu_vid = 0x045E;
static const int mcu_pid = 0x066F;

// Myriad X
static const int mx_vid = 0x03E7;
static const int mx_pid = 0x2485;


bool open_device(void)
{
    libusb_context *context = NULL;
    int result = libusb_init(&context);
    if ( result != 0){
        util::log_error("cannot open any usb");
        return false;
    }

    if (!libusb_open_device_with_vid_pid(NULL, mx_vid, mx_pid))
    {
        util::log_info("libusb_open_device_with_vid_pid VID 0x" + util::to_hex_string(mx_vid) + " PID 0x" + util::to_hex_string(mx_pid) + " failed");
        return false;
    }
    else
    {
        util::log_info("libusb_open_device_with_vid_pid VID 0x" + util::to_hex_string(mx_vid) + " PID 0x" + util::to_hex_string(mx_pid) + " found");
        return true;
    }
}

void authenticate_device(void)
{
    util::log_info("starting validator with VID 0x" + util::to_hex_string(mcu_vid) + " PID 0x" + util::to_hex_string(mcu_pid));
    start_validator(mcu_vid, mcu_pid);

    // wait for authentication
    while (true)
    {
        util::log_info("authentication status: " + std::to_string(check_som_status()));

        if (0 == check_som_status())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else
        {
            break;
        }
    }
}

} // namespace device
