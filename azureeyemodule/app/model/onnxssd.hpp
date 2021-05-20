// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// TODO: Merge this class with SSD.
#pragma once

// Standard library includes
#include <string>
#include <vector>

// Local includes
#include "objectdetector.hpp"

namespace model {

/**
 * A class to represent a ONNXSSD model on the Azure Percept device.
 */
class ONNXSSDModel : public ObjectDetector
{
public:
    ONNXSSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd,
                 const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    void run(cv::GStreamingCompiled* pipeline) override;

protected:
    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    virtual bool pull_data(cv::GStreamingCompiled &pipeline);

private:
    std::string onnxfpath;

    /** Compile the pipeline graph for SSD. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
