// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/render.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace cv {
namespace gapi {
namespace custom {

G_API_OP(BBoxes, <cv::GArray<cv::gapi::wip::draw::Prim>(cv::GArray<cv::Rect>)>, "sample.custom.b-boxes")
{
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &)
    {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes)
{
    // Converts the rectangles into G-API's rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs, std::vector<cv::gapi::wip::draw::Prim> &out_prims)
    {
        out_prims.clear();

        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr)
        {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };

        for (auto &&rc : in_face_rcs)
        {
            out_prims.emplace_back(cvt(rc, CV_RGB(0, 255, 0)));   // green
        }
    }
};

} // namespace custom
} // namesapace gapi
} // namespace cv
