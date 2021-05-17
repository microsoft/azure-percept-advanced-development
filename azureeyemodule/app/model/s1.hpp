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

class S1Model : public ObjectDetector
{
public:
    S1Model(const std::string &labelfpath, const std::vector<std::string> &modelfpaths,
            const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution, const std::string &json_configuration);

    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** We filter out all detected objects that are less than this value in confidence. */
    double confidence_threshold;

    /** We reject all detections whose bounding boxes overlap with higher-confidence boxes if they overlap by at least this much. */
    double nms_threshold;

    /** Compile the pipeline graph for S1. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
