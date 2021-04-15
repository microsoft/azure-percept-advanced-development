// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_kernels.hpp"


namespace cv {
namespace gapi {
namespace custom {


GDetectionsWithConf parse_yolo_with_confidences(const GMat& in, const GOpaque<Size>& in_sz, float confidence_threshold, float nms_threshold, const GYoloAnchors& anchors)
{
    return GParseYoloWithConf::on(in, in_sz, confidence_threshold, nms_threshold, anchors);
}

} // namespace custom
} // namespace gapi
} // namespace cv
