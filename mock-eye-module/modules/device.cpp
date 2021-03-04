// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard Library includes
#include <iostream>
#include <string>

// Local includes
#include "device.hpp"

namespace device {

Device look_up_device(const std::string &device_str)
{
    if (device_str == "CPU")
    {
        return Device::CPU;
    }
    else if (device_str == "GPU")
    {
        return Device::GPU;
    }
    else if ((device_str == "MYRIAD") || (device_str == "NCS2") || (device_str == "VPU"))
    {
        return Device::NCS2;
    }
    else
    {
        std::cerr << "Given " << device_str << " for --device, but we do not support it." << std::endl;
        exit(__LINE__);
    }
}

std::string device_to_string(const Device &device)
{
    switch (device)
    {
        case Device::CPU:
            return std::string("CPU");
        case Device::GPU:
            return std::string("GPU");
        case Device::NCS2:
            return std::string("MYRIAD");
        default:
            std::cerr << "Cannot convert the device to a string. The device enum must have been updated, but the string representations have not been." << std::endl;
            exit(__LINE__);
    }
}

} // namespace device
