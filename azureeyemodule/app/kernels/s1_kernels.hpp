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

using GDetectionsWithConf = std::tuple<GArray<Rect>, GArray<int>, GArray<float>>;

/** S1 Op */
G_API_OP(GParseS1WithConf, <GDetectionsWithConf(GMat, GMat, GOpaque<Size>, float, float)>, "org.opencv.dnn.parseS1WithConf")
{
    static std::tuple<GArrayDesc, GArrayDesc, GArrayDesc> outMeta(const GMatDesc&, const GMatDesc&, const GOpaqueDesc&, float, float)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc());
    }
};

/** Kernel implementation of S1 op */
GAPI_OCV_KERNEL(GOCVParseS1WithConf, GParseS1WithConf)
{
    static void run(const Mat & in_raw_boxes,
        const Mat & in_raw_probs,
        const Size & in_size,
        float confidence_threshold,
        float nms_threshold,
        std::vector<Rect> & out_boxes,
        std::vector<int> & out_labels,
        std::vector<float> & out_confidences)
    {
        const auto& in_boxes_dims = in_raw_boxes.size;
        GAPI_Assert(in_boxes_dims.dims() == 4u);

        const auto& in_probs_dims = in_raw_probs.size;
        GAPI_Assert(in_probs_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_probs_dims[2];
        const int NUM_CLASSES = in_probs_dims[3];
        const int OBJECT_SIZE = in_boxes_dims[3];

        out_boxes.clear();
        out_labels.clear();
        out_confidences.clear();

        struct Detection {
            cv::Rect rect;
            float    conf;
            int      label;
        };
        std::vector<Detection> detections;

        const auto boxes = in_raw_boxes.ptr<float>();
        const auto probs = in_raw_probs.ptr<float>();

        for (int i = 0; i < MAX_PROPOSALS; i++)
        {
            for (int label = 1; label < NUM_CLASSES; label++)
            {
                float confidence = probs[i * NUM_CLASSES + label];

                if (confidence < confidence_threshold)
                {
                    continue; // skip objects with low confidence
                }

                float center_x = boxes[i * OBJECT_SIZE];
                float center_y = boxes[i * OBJECT_SIZE + 1];
                float w = boxes[i * OBJECT_SIZE + 2];
                float h = boxes[i * OBJECT_SIZE + 3];

                const Rect surface({ 0,0 }, in_size);

                Rect rc;  // map relative coordinates to the original image scale
                rc.x = static_cast<int>((center_x - w / 2) * in_size.width);
                rc.y = static_cast<int>((center_y - h / 2) * in_size.height);
                rc.width = static_cast<int>(w * in_size.width);
                rc.height = static_cast<int>(h * in_size.height);

                detections.emplace_back(Detection{ rc, confidence, label - 1 });
            }
        }

        std::stable_sort(std::begin(detections), std::end(detections),
            [](const Detection& a, const Detection& b) {
                return a.conf > b.conf;
            });

        if (nms_threshold < 1.0f)
        {
            for (const auto& d : detections)
            {
                // Reject boxes which overlap with previously pushed ones
                // (They are sorted by confidence, so rejected box
                // always has a smaller confidence
                if (std::end(out_boxes) ==
                    std::find_if(std::begin(out_boxes), std::end(out_boxes),
                        [&d, nms_threshold](const Rect& r) {
                            float rectOverlap = 1.f - static_cast<float>(jaccardDistance(r, d.rect));
                            return rectOverlap > nms_threshold;
                        }))
                {
                    out_boxes.emplace_back(d.rect);
                    out_labels.emplace_back(d.label);
                    out_confidences.emplace_back(d.conf);
                }
            }
        }
        else
        {
            for (const auto& d : detections)
            {
                out_boxes.emplace_back(d.rect);
                out_labels.emplace_back(d.label);
                out_confidences.emplace_back(d.conf);
            }
        }
    }
};

/** C++ wrapper for S1 parsing */
GAPI_EXPORTS GDetectionsWithConf parseS1WithConf(const GMat& in_raw_boxes, const GMat& in_raw_probs, const GOpaque<Size>& in_sz, float confidence_threshold = 0.5f, float nms_threshold = 0.5f);

} // namespace streaming
} // namespace gapi
} // namespace cv
