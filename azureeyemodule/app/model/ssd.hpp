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
    SSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd,
             const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution, const std::string &json_configuration);

    /**
     * The SSD model acts as our default model in case we don't get direction
     * one way or another or need to fall back to a working model.
     */
    void load_default();

    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** We filter out all detected objects that are less than this value in confidence. */
    double confidence_threshold;

    /** If not -1, we filter out all detections except the class matching this number. */
    int filter_label;

    /** Compile the pipeline graph for SSD. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
