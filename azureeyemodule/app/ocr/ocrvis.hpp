// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Derived from some demo code found here: https://github.com/openvinotoolkit/open_model_zoo
// And taken under Apache License 2.0
// Copyright (C) 2020 Intel Corporation
#pragma once

// Standard library includes
#include <vector>

// Third-party includes
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/mx.hpp> // size()

namespace ocr {
namespace vis {
    // Draw rotated rectangles on the texts detected on an Image scene
    void drawRotatedRect(cv::Mat &m, const cv::RotatedRect &rc);

    // Put texts and textboxes on the recognized TextLabels
    void drawText(cv::Mat &m, const cv::RotatedRect &rc, const std::string &str);

} //namespace vis
} // namespace ocr
