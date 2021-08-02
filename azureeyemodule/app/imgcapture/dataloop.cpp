// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <dirent.h>
#include <errno.h>
#include <string>
#include <thread>

// Third party includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

// Local includes
#include "dataloop.hpp"
#include "../secure_ai/secureai.hpp"
#include "../util/helper.hpp"

namespace loop {

/** The directory in which we save the files for periodic uploading */
static const std::string snapshot_dpath = "/app/snapshot/";

/** (Re)initialize the storage directory. */
static void create_snapshot_storage()
{
    int ret = util::run_command("rm -rf " + snapshot_dpath + " && mkdir " + snapshot_dpath);
    if (ret != 0)
    {
        util::log_error("rm && mkdir failed with " + ret);
    }
}

static void upload_snapshot_file(const std::string &fpath, const secure::SecureAIParams &security_params, const std::string &confidence)
{
    if (security_params.secure_ai_lifecycle_enabled)
    {
        int ret = util::run_command(("python3 /usr/lib/python3.7/site-packages/sczpy.py " + security_params.model_management_server_url + " export " + security_params.model_name + " " + security_params.model_version + " " + confidence + " " + fpath).c_str());
        if (ret != 0)
        {
            util::log_error("sczpy.py failed when trying to upload a file for the retraining loop. Error code: " + std::to_string(ret) + ", file: " + fpath);
        }
    }
    else
    {
        // TODO
        util::log_error("Uploading retraining data currently requires secure AI to be enabled.");
    }

    // Either way, remove the file to prevent cluttering up non-volatile storage with images
    std::remove(fpath.c_str());
}

static void upload_snapshot_directory(const secure::SecureAIParams &security_params)
{
    DIR *d_data;

    d_data = opendir(snapshot_dpath.c_str());
    if (d_data == NULL)
    {
        util::log_error("Could not open snapshot directory for retrain loop data upload. Errno: " + std::to_string(errno));
        return;
    }

    // Try to upload each file in the directory
    struct dirent *f_data;
    while ((f_data = readdir(d_data)) != NULL)
    {
        std::string filename(f_data->d_name);

        // Only upload the files that have sensible names (need a confidence score)
        size_t pos = filename.find_first_of("-");
        if (pos != std::string::npos)
        {
            std::string confidence = filename.substr(0, pos);
            auto fpath = snapshot_dpath + filename;

            upload_snapshot_file(fpath, security_params, confidence);
        }
    }

    closedir(d_data);
}

void write_data_file(const std::string &filename, const cv::Mat &bgr)
{
    cv::imwrite(snapshot_dpath + filename, bgr);
}

void *export_data(void *)
{
    create_snapshot_storage();

    while (true)
    {
        auto security_params = secure::get_model_params();
        upload_snapshot_directory(security_params);

        // TODO: I don't like this hard-coded sleep. What if uploading takes too long
        //       and we end up not removing images as fast as we are saving them?
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

} // namespace loop
