// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/streaming/desync.hpp>

namespace cv {
namespace gapi {
namespace streaming {

/** A classifier returns a one-hot integer label ID vector and the confidences. */
using GClassificationsWithConf = std::tuple<GArray<int>, GArray<float>>;

/** Classification op */
G_API_OP(GParseClass, <GClassificationsWithConf(GMat)>, "org.opencv.dnn.parseClass")
{
    static std::tuple<GArrayDesc, GArrayDesc> outMeta(const GMatDesc&)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc());
    }
};

/** Kernel implementation of classification op */
GAPI_OCV_KERNEL(GOCVParseClass, GParseClass)
{
    static void run(const Mat &in_result, std::vector<int> &out_labels, std::vector<float> &out_confidences)
    {
        const auto& in_dims = in_result.size;

        // We expect a Tensor of shape (1, 1, 1, N)
        CV_Assert(in_dims.dims() == 4);

        out_labels.clear();
        out_confidences.clear();

        const auto results = in_result.ptr<float>();
        for (int i = 0; i < in_dims[3]; i++)
        {
            out_labels.emplace_back(i);
            out_confidences.emplace_back(results[i]);
        }
    }
};

/** C++ wrapper for classification op */
GAPI_EXPORTS GClassificationsWithConf parse_class(const GMat& in);

} // namespace streaming
} // namespace gapi
} // namespace cv