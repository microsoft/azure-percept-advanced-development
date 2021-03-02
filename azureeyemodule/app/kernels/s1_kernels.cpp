// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/streaming/desync.hpp>

#include "s1_kernels.hpp"

namespace cv {
namespace gapi {
namespace streaming {

/** C++ wrapper for S1 parsing */
GDetectionsWithConf parseS1WithConf(const GMat& in_raw_boxes, const GMat& in_raw_probs, const GOpaque<Size>& in_sz, float confidence_threshold, float nms_threshold)
{
    return GParseS1WithConf::on(in_raw_boxes, in_raw_probs, in_sz, confidence_threshold, nms_threshold);
};

} // namespace streaming
} // namespace gapi
} // namespace cv
