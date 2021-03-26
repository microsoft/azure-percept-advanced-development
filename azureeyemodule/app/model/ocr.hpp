// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// standard library includes
#include <string>
#include <vector>
#include <map>
#include <thread>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/streaming/desync.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "../ocr/decoder.hpp"
#include "../ocr/ocrvis.hpp"

namespace model {

/** A class to represent a OCR Model : Text Detection & Text Recognition Model on the Azure Percept Device */
class OCRModel : public AzureEyeModel
{
public:
    OCRModel( const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    void run(cv::GStreamingCompiled* pipeline) override;

private:

    /** OCRDecoder is our TextDecoder which will run the CTCGreedy/ Beamsearch */
    ocr::TextDecoder OCRDecoder;

    /** Compile the graph for classification model */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    bool pull_data(cv::GStreamingCompiled &pipeline);

    /** Update the BGR output to show the labels. */
    void handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<cv::RotatedRect> &last_rcs, const std::vector<std::string> &last_text);

    /** Compose the RGB based on the labels. */
    void preview(cv::Mat &bgr, const std::vector<cv::RotatedRect> &last_rcs, const std::vector<std::string> &last_text) const;

    /** Handle the detector's output */
    void handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                        const cv::optional<std::vector<cv::RotatedRect>> &out_txtrcs, std::vector<cv::RotatedRect> &last_rcs,
                                        cv::optional<std::vector<cv::Mat>> &out_text, std::vector<std::string> &last_text);

    /** Print out all the model's meta information **/
    void log_parameters() const;
};

} // namespace model
