// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>

// Local includes
#include "azureeyemodel.hpp"
namespace model {

/**
 * A class to represent a binary unet model on the Azure Percept device.
 */
class BinaryUnetModel : public AzureEyeModel
{
public:
    BinaryUnetModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    void run(cv::GStreamingCompiled* pipeline) override;

private:

    /** Compile the graph for classification model */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    bool pull_data(cv::GStreamingCompiled &pipeline);

    /** Update the BGR output to show segmentation. */
    void handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const cv::Mat &last_mask);

    /** Actually draws segmented output. */
    void preview(cv::Mat& rgb, const cv::Mat& last_mask) const;

    /** Send the class percentage occupancy message */
    void handle_inference_output(const cv::optional<cv::Mat> &out_mask,
                                    cv::Mat &last_mask,
                                    float threshold = 0.5
                                    );

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
