// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "objectdetector.hpp"


namespace model {

/**
 * A class to represent a Single Shot Detector model for the Azure Percept device.
 */
class SSDModel : public ObjectDetector
{
public:
    SSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    /**
     * The SSD model acts as our default model in case we don't get direction
     * one way or another or need to fall back to a working model.
     */
    void load_default();

    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** Compile the pipeline graph for SSD. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
