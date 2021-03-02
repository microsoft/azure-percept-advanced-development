// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef __VALIDATOR_H__
#define __VALIDATOR_H__
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
    // Note: Validator only supports the scenario that one Ear/Eye SoM attached to the DevKit. Multiple SoM scenario isn't supported so far.

    // Start SoM authentication monitor.
    // This function initializes a new thread to detect SoM PNP events continuously. It kicks off authentication operation
    // immediately once a SoM device is detected, and releases the SoM device resource once the authenciation operation completes
    // or the SoM device is detached.
    // Params:
    // @som_vid: the vendor ID of the SoM board to be monitored
    // @som_pid: the production ID of the SoM board to be monitored
    // Returns:
    // True: The monitor thread is set up properly.
    // False: The monitor thread isn't set up properly.
    bool start_validator(uint16_t som_vid, uint16_t som_pid);

    // Get current SoM authentication status.
    // Returns:
    // True: authentic SOM device is present.
    // False: authentic SoM device isn't present
    bool check_som_status();

    // Stop SoM authentication monitor. It should be called when the main thread exits.
    int stop_validator();

    // Perform SoM authentication single operation and return authentication result.
    // Note: This function doesn't perform any SoM PNP detection. It only enumerates the current exiting SoM device and perform authentication for it.
    // This function is only for those advanced developers who perfers to maintain their own PNP detection logic in their application.
    // Params:
    // @som_vid: the vendor ID of the SoM board to be authenticated
    // @som_pid: the production ID of the SoM board to be authenticated
    // Returns:
    // True: The SOM device is authentic.
    // False: The SOM device isn't authentic.
    bool start_som_auth(uint16_t som_vid, uint16_t som_pid);

#ifdef __cplusplus
}
#endif

#endif
