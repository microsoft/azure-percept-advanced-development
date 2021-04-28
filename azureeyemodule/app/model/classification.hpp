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
 * A class to represent a classification model on the Azure Percept device.
 */
class ClassificationModel : public AzureEyeModel
{
public:
    ClassificationModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** Path to the label file. */
    std::string labelfpath;

    /** Labels for the things we classify. */
    std::vector<std::string> class_labels;

    /** Compile the graph for classification model */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    bool pull_data(cv::GStreamingCompiled &pipeline);

    /** Update the BGR output to show the labels. */
    void handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<int> &last_labels, const std::vector<float> &last_confidences);

    /** Compose the RGB based on the labels. */
    void preview(const cv::Mat& rgb, const std::vector<int>& labels, const std::vector<float>& confidences) const;

    /** Handle the detector's output */
    void handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                                      const cv::optional<std::vector<int>> &out_labels,
                                                      const cv::optional<std::vector<float>> &out_confidences, std::vector<int> &last_labels,
                                                      std::vector<float> &last_confidences);

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
