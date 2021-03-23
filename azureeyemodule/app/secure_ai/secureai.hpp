// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include <string>

namespace secure {

/** A simple bag of data items that we keep together for the secure AI stuff. */
struct SecureAIParams {
    /** Is our secure AI lifecycle stuff enabled? */
    bool secure_ai_lifecycle_enabled = false;

    /** Is the model file downloaded from model management server? */
    bool download_from_model_management_server = true;

    /** Name of the model */
    std::string model_name = "";

    /** Version of the model */
    std::string model_version = "";

    /** Model management server URL */
    std::string model_management_server_url = "";

    /** External model file URL */
    std::string model_url = "";

    /** Returns a string representation of this struct. */
    std::string to_string() const;

    /**
     * Create a SecureAIParams struct from the given string.
     *
     * Attempts to do this based on the values in `str`.
     *
     * This method oparates by doing the inverse of SecureAIParams::to_string(),
     * and so is best suited for deserializing from a string that was created
     * by that method.
     *
     * If any fields cannot be determined, we log an error and return false.
     */
    static bool from_string(const std::string &str, SecureAIParams &result);
};

/**
 * Update the secure AI lifecycle parameters.
 * Returns true if anything is different from what we had before.
 *
 * NOTE: This operates using a mutex and blocks until it is available.
 */
bool update_secure_model_params(const std::string &model_managment_server_url, const std::string &model_name,
                                const std::string &model_version, bool enable, bool download_from_model_management_server,
                                const std::string &model_url);

/**
 * Get a copy of the current secure AI model parameters.
 *
 * NOTE: This is thread-safe, as it creates a copy and the creation of the copy is hidden behind a mutex. We block until
 *       we get that mutex.
 */
SecureAIParams get_model_params();

/**
 * Download the model from model management server or shared URL, and then decrypt it with sczpy APIs.
 * Returns true on success.
 *
 * This function does NOT take the mutex, as it does not access the singleton secure AI parameters.
 */
bool download_model(std::vector<std::string> &modelfiles, const std::string &secureconfig);

} // namespace secure
