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
static const int MCU_VID = 0x045E;
static const int MCU_PID = 0x066F;

// Myriad X
static const int MX_VID = 0x03E7;
static const int MX_PID = 0x2485;


bool open_device(void)
{
    int result = libusb_init(nullptr);
    if (result != LIBUSB_SUCCESS)
    {
        util::log_error("Cannot initialize libusb. Error code: " + std::to_string(result));
        return false;
    }

    // Get all USB devices connected to the system
    libusb_device **devlist = nullptr;
    size_t n_usb_items = libusb_get_device_list(nullptr, &devlist);
    if (n_usb_items == 0)
    {
        util::log_error("No USB devices found.");
        return false;
    }

    // Check each connected USB device's VID and PID against the Eye SOM's
    std::vector<size_t> matching_descriptors;
    for (size_t i = 0; i < n_usb_items; i++)
    {
        libusb_device *usb_device = devlist[i];
        libusb_device_descriptor descriptor;
        result = libusb_get_device_descriptor(usb_device, &descriptor);
        if (result != LIBUSB_SUCCESS)
        {
            util::log_warning("Cannot retrieve device descriptor for device index " + std::to_string(i) + " in libusb_get_device_list.");
            continue;
        }

        if ((descriptor.idVendor == MX_VID) && (descriptor.idProduct == MX_PID))
        {
            matching_descriptors.push_back(i);
        }
    }

    if (matching_descriptors.size() == 0)
    {
        util::log_warning("Found " + std::to_string(n_usb_items) + " connected USB devices, but none of them have VID 0x" + util::to_hex_string(MX_VID) + " and PID 0x" + util::to_hex_string(MX_PID));
        matching_descriptors.clear();
        libusb_free_device_list(devlist, (int)true);
        return false;
    }
    else if (matching_descriptors.size() > 1)
    {
        util::log_error("Expecting to find only one, but found " + std::to_string(matching_descriptors.size()) + " descriptors with VID 0x" + util::to_hex_string(MX_VID) + " and PID 0x" + util::to_hex_string(MX_PID));
        matching_descriptors.clear();
        libusb_free_device_list(devlist, (int)true);
        return false;
    }

    assert(matching_descriptors.size() == 1);

    // Get the device from the list (there should only be one device in the list)
    size_t devidx = matching_descriptors[0];
    libusb_device *usb_device = devlist[devidx];

    // Open the device (and free the list)
    libusb_device_handle *handle;
    result = libusb_open(usb_device, &handle);
    libusb_free_device_list(devlist, (int)true);
    if (result != LIBUSB_SUCCESS)
    {
        switch (result)
        {
            case LIBUSB_ERROR_NO_MEM:
                util::log_error("Cannot open Eye SOM: memory allocation failure in libusb.");
                return false;
            case LIBUSB_ERROR_ACCESS:
                util::log_error("Cannot open Eye SOM: insufficient permissions.");
                return false;
            case LIBUSB_ERROR_NO_DEVICE:
                util::log_error("Cannot open Eye SOM: Device has been disconnected.");
                return false;
            default:
                util::log_error("Cannot open Eye SOM: unknown error in libusb open. Error code: " + std::to_string(result));
                return false;
        }
    }

    return true;
}

void authenticate_device(void)
{
    util::log_info("starting validator with VID 0x" + util::to_hex_string(MCU_VID) + " PID 0x" + util::to_hex_string(MCU_PID));
    start_validator(MCU_VID, MCU_PID);

    // wait for authentication
    while (true)
    {
        bool som_success = check_som_status();
        util::log_info("authentication status: " + std::to_string(som_success));

        if (som_success)
        {
            break;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

} // namespace device
