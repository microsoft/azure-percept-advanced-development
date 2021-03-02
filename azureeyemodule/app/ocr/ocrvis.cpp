// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Derived from some demo code found here: https://github.com/openvinotoolkit/open_model_zoo
// And taken under Apache License 2.0
// Copyright (C) 2020 Intel Corporation

// Standard library includes
#include <vector>

// Third-party includes
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/mx.hpp> // size()

// Local files includes
#include "ocrvis.hpp"
#include "../kernels/ocr_kernels.hpp"

namespace ocr {
namespace vis {

void drawRotatedRect(cv::Mat &m, const cv::RotatedRect &rc)
{
    std::vector<cv::Point2f> tmp_points(5);
    rc.points(tmp_points.data());
    tmp_points[4] = tmp_points[0];
    auto prev = tmp_points.begin(), it = prev+1;
    for (; it != tmp_points.end(); ++it)
    {
        cv::line(m, *prev, *it, cv::Scalar(50, 205, 50), 2);
        prev = it;
    }
};

void drawText(cv::Mat &m, const cv::RotatedRect &rc, const std::string &str)
{
    const int    fface   = cv::FONT_HERSHEY_SIMPLEX;
    const double scale   = 0.7;
    const int    thick   = 1;
          int    base    = 0;
    const auto text_size = cv::getTextSize(str, fface, scale, thick, &base);

    std::vector<cv::Point2f> tmp_points(4);
    rc.points(tmp_points.data());
    const auto tl_point_idx = cv::gapi::streaming::OCVCropLabels::topLeftPointIdx(tmp_points);
    cv::Point text_pos = tmp_points[tl_point_idx];
    text_pos.x = std::max(0, text_pos.x);
    text_pos.y = std::max(text_size.height, text_pos.y);

    cv::rectangle(m, text_pos + cv::Point{0, base},
                  text_pos + cv::Point{text_size.width, -text_size.height},
                  CV_RGB(50, 205, 50),
                  cv::FILLED);
    const auto white = CV_RGB(255, 255, 255);
    cv::putText(m, str, text_pos, fface, scale, white, thick, 8);
};

} // namespace vis
} //namespace ocr
