// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "azureeyemodel.hpp"


namespace model {

/**
 * An abstract class that derives from the AzureEyeModel base class
 * and which represents an Object Detector, such as SSD or YOLO.
 */
class ObjectDetector : public AzureEyeModel
{
public:
    ObjectDetector(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);
    virtual ~ObjectDetector();

protected:
    /** Path to the label file. */
    std::string labelfpath;

    /** Labels for the things we classify. */
    std::vector<std::string> class_labels;

    /** Update the BGR output to show the bounding boxes and labels. */
    virtual void handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<cv::Rect> &last_boxes, const std::vector<int> &last_labels, const std::vector<float> &last_confidences);

    /** Handle the detector's output */
    virtual void handle_inference_output(const cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                         const cv::optional<std::vector<cv::Rect>> &out_boxes, const cv::optional<std::vector<int>> &out_labels,
                                         const cv::optional<std::vector<float>> &out_confidences, const cv::optional<cv::Size> &out_size, std::vector<cv::Rect> &last_boxes, std::vector<int> &last_labels,
                                         std::vector<float> &last_confidences);

    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    virtual bool pull_data(cv::GStreamingCompiled &pipeline);

private:
    /** Compose the RGB based on the labels and boxes. */
    void preview(const cv::Mat &rgb, const std::vector<cv::Rect> &boxes, const std::vector<int> &labels, const std::vector<float> &confidences) const;
};

} // namespace model
