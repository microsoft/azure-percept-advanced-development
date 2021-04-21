// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * Utility ops and kernels.
 */
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <exception>

namespace cv {
namespace gapi {
namespace custom {

G_API_OP(GSize, <GOpaque<Size>(GMat)>, "org.opencv.util.size")
{
    static GOpaqueDesc outMeta(const GMatDesc&)
    {
        return empty_gopaque_desc();
    }
};

G_API_OP(GSizeR, <GOpaque<Size>(GOpaque<Rect>)>, "org.opencv.util.sizeR")
{
    static GOpaqueDesc outMeta(const GOpaqueDesc&)
    {
        return empty_gopaque_desc();
    }
};

GAPI_EXPORTS GOpaque<Size> size(const GMat& src);
GAPI_EXPORTS GOpaque<Size> size(const GOpaque<Rect>& r);

GAPI_OCV_KERNEL(GOCVSize, cv::gapi::custom::GSize) {
    static void run(const cv::Mat& in, cv::Size& out) {
        out.width  = in.cols;
        out.height = in.rows;
    }
};

GAPI_OCV_KERNEL(GOCVSizeR, cv::gapi::custom::GSizeR) {
    static void run(const cv::Rect& in, cv::Size& out) {
        out.width  = in.width;
        out.height = in.height;
    }
};

} // custom namespace
} // gapi namespace
} // cv namespace
