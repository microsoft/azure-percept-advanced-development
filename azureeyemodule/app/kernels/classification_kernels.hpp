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
G_API_OP(GParseClass, <GClassificationsWithConf(GMat, GOpaque<Size>, float)>, "org.opencv.dnn.parseClass")
{
    static std::tuple<GArrayDesc, GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&, float)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc());
    }
};

/** Kernel implementation of classification op */
GAPI_OCV_KERNEL(GOCVParseClass, GParseClass)
{
    static void run(const Mat & in_result, const Size & in_size, float confidence_threshold, std::vector<int> & out_labels, std::vector<float> & out_confidences)
    {
        const auto& in_dims = in_result.size;

        out_labels.clear();
        out_confidences.clear();

        const auto results = in_result.ptr<float>();
        float max_confidence = 0;
        int label = 0;

        for (int i = 0; i < in_dims[1]; i++)
        {
            if (results[i] > max_confidence)
            {
                label = i;
                max_confidence = results[label];
            }
        }

        out_labels.emplace_back(label);
        out_confidences.emplace_back(results[label]);
    }
};

/** C++ wrapper for classification op */
GAPI_EXPORTS GClassificationsWithConf parseClass(const GMat& in, const GOpaque<Size>& in_sz, float confidence_threshold = 0.5f);

} // namespace streaming
} // namespace gapi
} // namespace cv
