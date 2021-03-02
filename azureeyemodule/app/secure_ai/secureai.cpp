// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <mutex>
#include <string>
#include <sstream>
#include <vector>

// Local includes
#include "secureai.hpp"
#include "../util/helper.hpp"

namespace secure {

/** The single secure params that we keep track of */
static SecureAIParams secure_params;

/** Grab this whenever you are doing any operations with the secure_params. */
static std::mutex secure_params_mutex;


bool update_secure_model_params(const std::string &model_managment_server_url, const std::string &name, const std::string &version, bool enable, bool download_from_model_management_server, const std::string &model_url)
{
    secure_params_mutex.lock();
    bool all_the_same = false;

    // If secure_ai_lifecycle_enabled keeps disabled, then we only check the model_url change.
    if (!enable
        && (secure_params.secure_ai_lifecycle_enabled == enable)
        && (secure_params.model_url == model_url))
    {
        all_the_same = true;
    }
    // If secure_ai_lifecycle_enabled and download_from_model_management_server keeps enabled, then we ignore the model_url change.
    else if (enable
            && download_from_model_management_server
            && (secure_params.secure_ai_lifecycle_enabled == enable)
            && (secure_params.model_name == name)
            && (secure_params.model_version == version)
            && (secure_params.model_management_server_url == model_managment_server_url)
            && (secure_params.download_from_model_management_server == download_from_model_management_server))
    {
        all_the_same = true;
    }
    else
    {
        all_the_same = (secure_params.model_name == name)
                    && (secure_params.model_version == version)
                    && (secure_params.model_management_server_url == model_managment_server_url)
                    && (secure_params.secure_ai_lifecycle_enabled == enable)
                    && (secure_params.download_from_model_management_server == download_from_model_management_server)
                    && (secure_params.model_url == model_url);
    }

    secure_params.model_name = name;
    secure_params.model_version = version;
    secure_params.model_management_server_url = model_managment_server_url;
    secure_params.secure_ai_lifecycle_enabled = enable;
    secure_params.download_from_model_management_server = download_from_model_management_server;
    secure_params.model_url = model_url;
    secure_params_mutex.unlock();

    return !all_the_same;
}

SecureAIParams get_model_params()
{
    secure_params_mutex.lock();
    auto params = secure_params; // Copy the struct, field by field (by value)
    secure_params_mutex.unlock();

    return secure_params;
}


// Note that this method needs to mirror from_string as it serves as a serializer
std::string SecureAIParams::to_string() const
{
    std::stringstream str;
    str << this->model_management_server_url << "," << this->model_name << "," << this->model_version << "," << (this->secure_ai_lifecycle_enabled ? "true" : "false") << "," << (this->download_from_model_management_server ? "true" : "false") << "," << this->model_url;
    return str.str();
}

// Note that this method needs to mirror to_string, as it serves as a deserializer
bool SecureAIParams::from_string(const std::string &str, SecureAIParams &result)
{
    std::vector<std::string> our_fields = {"model_managment_server_url", "name", "version", "enabled", "download_from_model_management_server", "model_url"};
    std::vector<std::string> fields = util::splice_comma_separated_list(str);
    if (fields.size() != our_fields.size())
    {
        return false;
    }

    result.model_management_server_url = fields.at(0);
    result.model_name = fields.at(1);
    result.model_version = fields.at(2);
    result.secure_ai_lifecycle_enabled = (fields.at(3) == "true");
    result.download_from_model_management_server = (fields.at(4) == "true");
    result.model_url = fields.at(5);

    return true;
}

/** Try to download the model and decrypt it using sczpy.py. Return whether we succeeded or not. */
bool download_model(std::vector<std::string> &modelfiles, const std::string &secureconfig)
{
    secure::SecureAIParams secure_ai_params;
    bool worked = secure::SecureAIParams::from_string(secureconfig, secure_ai_params);
    if (!worked)
    {
        util::log_error("Could not deserialize the secure AI parameters. Given a string that doesn't make sense: " + secureconfig);
        return false;
    }

    // The script should download the model to this location
    const std::string encryptedmodelpath = "/app/model/model.enc.zip";
    // The script should decrypt the model to this location
    const std::string decryptedmodelpath = "/app/model/model.dec.zip";
    int ret;

    if (secure_ai_params.download_from_model_management_server)
    {
        // Run the sczpy download command and see if it worked
        ret = util::run_command(("python3 /usr/lib/python3.7/site-packages/sczpy.py " + secure_ai_params.model_management_server_url + " download " + secure_ai_params.model_name + " " + secure_ai_params.model_version + " " + encryptedmodelpath + " " + decryptedmodelpath).c_str());
        if (ret != 0)
        {
            util::log_error("sczpy.py failed when trying to download the model file. Error code: " + std::to_string(ret));
            return false;
        }
    }
    else
    {
        ret = util::run_command(("wget --no-check-certificate -O " + encryptedmodelpath + " \"" + secure_ai_params.model_url + "\"").c_str());
        if (ret != 0)
        {
            util::log_error("wget failed with " + ret);
            return false;
        }

        ret = util::run_command(("python3 /usr/lib/python3.7/site-packages/sczpy.py " + secure_ai_params.model_management_server_url + " decrypt " + secure_ai_params.model_name + " " + secure_ai_params.model_version + " " + encryptedmodelpath + " " + decryptedmodelpath).c_str());
        if (ret != 0)
        {
            util::log_error("sczpy.py failed when trying to decrypt the model file. Error code: " + std::to_string(ret));
            return false;
        }
    }

    modelfiles = {decryptedmodelpath};

    return true;
}
} // namespace secure
