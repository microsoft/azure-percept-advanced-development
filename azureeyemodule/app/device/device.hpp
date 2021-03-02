/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This module provides a high-level driver API for the Myriad X Eye SOM device.
 */
#pragma once

namespace device {

/**
 * Open the Eye SOM over USB.
 *
 * Returns true if successfully opened, false if we fail to open it.
 */
bool open_device(void);

/**
 * Authenticate the device.
 *
 * This function blocks until authentication is complete.
 */
void authenticate_device(void);

} // namespace device
