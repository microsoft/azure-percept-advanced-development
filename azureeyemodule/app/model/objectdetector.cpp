// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <string>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "objectdetector.hpp"
#include "../iot/iot_interface.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace model {

ObjectDetector::ObjectDetector(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, labelfpath(labelfpath), class_labels({})
{
}

ObjectDetector::~ObjectDetector()
{
}

void ObjectDetector::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, cv::Mat &last_bgr, const std::vector<cv::Rect> &last_boxes,
                                       const std::vector<int> &last_labels, const std::vector<float> &last_confidences)
{
    // BGR output: visualize and optionally display
    if (!out_bgr.has_value())
    {
        return;
    }

    util::log_debug("bgr: size=" + util::to_size_string(out_bgr.value()));

    last_bgr = *out_bgr;

    cv::Mat original_bgr;
    last_bgr.copyTo(original_bgr);

    rtsp::update_data_raw(last_bgr);
    preview(last_bgr, last_boxes, last_labels, last_confidences);

    if (this->status_msg.empty())
    {
        rtsp::update_data_result(last_bgr);
    }
    else
    {
        cv::Mat bgr_with_status;
        last_bgr.copyTo(bgr_with_status);

        util::put_text(bgr_with_status, this->status_msg);
        rtsp::update_data_result(bgr_with_status);
    }

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(original_bgr, last_confidences);
}

void ObjectDetector::preview(const cv::Mat& rgb, const std::vector<cv::Rect>& boxes, const std::vector<int>& labels, const std::vector<float>& confidences) const
{
    for (std::size_t i = 0; i < boxes.size(); i++)
    {
        // color of a label
        int index = labels[i] % label::colors().size();

        cv::rectangle(rgb, boxes[i], label::colors().at(index), 2);
        cv::putText(rgb,
            util::get_label(labels[i], this->class_labels) + ": " + util::to_string_with_precision(confidences[i], 2),
            boxes[i].tl() + cv::Point(3, 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(label::colors().at(index)),
            2);
    }
}

void ObjectDetector::handle_inference_output(const cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                              const cv::optional<std::vector<cv::Rect>> &out_boxes, const cv::optional<std::vector<int>> &out_labels,
                                              const cv::optional<std::vector<float>> &out_confidences, const cv::optional<cv::Size> &out_size, std::vector<cv::Rect> &last_boxes, std::vector<int> &last_labels,
                                              std::vector<float> &last_confidences) const
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // The below objects are on the same desynchronized path
    // and are coming together
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_boxes.has_value());
    CV_Assert(out_labels.has_value());
    CV_Assert(out_confidences.has_value());
    CV_Assert(out_size.has_value());

    // Compose a message for each item we detected
    std::vector<std::string> messages;
    for (std::size_t i = 0; i < out_labels->size(); i++)
    {
        std::string str = std::string("{");

        cv::Rect rect = out_boxes.value()[i];
        cv::Rect2f rect_abs(static_cast<float>(rect.x) / out_size->width,
            static_cast<float>(rect.y) / out_size->height,
            static_cast<float>(rect.width) / out_size->width,
            static_cast<float>(rect.height) / out_size->height);

        static char buf[500];
        snprintf(buf, 500, "\"bbox\": [%.3f, %.3f, %.3f, %.3f]", rect_abs.x, rect_abs.y, rect_abs.x + rect_abs.width, rect_abs.y + rect_abs.height);
        str.append(buf)
            .append(",");

        str.append("\"label\": \"").append(util::get_label(out_labels.value()[i], this->class_labels)).append("\", ")
           .append("\"confidence\": \"").append(std::to_string(out_confidences.value()[i])).append("\", ")
           .append("\"timestamp\": \"").append(std::to_string(*out_nn_ts)).append("\"")
           .append("}");

        messages.push_back(str);
    }

    // Update our last items
    last_boxes = std::move(*out_boxes);
    last_labels = std::move(*out_labels);
    last_confidences = std::move(*out_confidences);

    // Compose a single string out of all the detection messages
    std::string str = std::string("[");
    for (size_t i = 0; i < messages.size(); i++)
    {
        if (i > 0)
        {
            str.append(", ");
        }
        str.append(messages[i]);
    }
    str.append("]");

    util::log_info("nn: seqno=" + std::to_string(*out_nn_seqno) + ", ts=" + std::to_string(*out_nn_ts) + ", " + str);

    // Send out the detection message to anyone who's listening
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, str);
}

bool ObjectDetector::pull_data(cv::GStreamingCompiled &pipeline)
{
    cv::optional<cv::Mat> out_bgr;

    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    cv::optional<cv::Mat> out_nn;
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<cv::Rect>> out_boxes;
    cv::optional<std::vector<int>> out_labels;
    cv::optional<std::vector<float>> out_confidences;
    cv::optional<cv::Size> out_size;

    std::vector<cv::Rect> last_boxes;
    std::vector<int> last_labels;
    std::vector<float> last_confidences;
    cv::Mat last_bgr;

    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_nn_seqno, out_nn_ts, out_boxes, out_labels, out_confidences, out_size)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_bgr, out_nn_ts, out_nn_seqno, out_boxes, out_labels, out_confidences, out_size, last_boxes, last_labels, last_confidences);
        this->handle_bgr_output(out_bgr, last_bgr, last_boxes, last_labels, last_confidences);

        if (this->restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

} // namespace model
