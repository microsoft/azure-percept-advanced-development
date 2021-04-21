
/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This module is responsible for the different types of devices we support
 * for this mock azureeyemodule application.
 *
 * For porting a new AI model, you don't need to worry about this file.
 */
#pragma once

#include <string>

namespace device
{

/** Backend device used for inference */
enum class Device {
    CPU,
    GPU,
    NCS2
};

/**
 * Look up the device based on the input argument.
 *
 * @param device_str The device in command-line string representation
 * @returns The enum representation of the device.
 */
Device look_up_device(const std::string &device_str);

/**
 * Convert a Device back into a string, suitable for backend interpretation by OpenVINO.
 *
 * @param device The device to convert to a string.
 * @return The string representation.
 */
std::string device_to_string(const Device &device);

} // namespace device
