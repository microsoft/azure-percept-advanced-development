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
using GYoloAnchors = std::vector<float>;

/** YOLO Op */
G_API_OP(GParseYoloWithConf, <GDetectionsWithConf(GMat, GOpaque<Size>, float, float, GYoloAnchors)>, "org.opencv.dnn.parseYoloWithConf")
{
    static std::tuple<GArrayDesc, GArrayDesc, GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&, float, float, const GYoloAnchors&)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc());
    }

    static const GYoloAnchors& defaultAnchors()
    {
        static GYoloAnchors anchors { 0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282,3.52778, 9.77052, 9.16828 };
        return anchors;
    }
};

namespace {
    class YoloParser
    {
        const float* m_out;
        int m_side, m_lcoords, m_lclasses;

        int index(int i, int b, int entry)
        {
            return b * m_side * m_side * (m_lcoords + m_lclasses + 1) + entry * m_side * m_side + i;
        }

    public:
        YoloParser(const float* out, int side, int lcoords, int lclasses)
            : m_out(out), m_side(side), m_lcoords(lcoords), m_lclasses(lclasses)
        {
            // Nothing to do
        }

        float scale(int i, int b)
        {
            int obj_index = index(i, b, m_lcoords);
            return m_out[obj_index];
        }

        double x(int i, int b)
        {
            int box_index = index(i, b, 0);
            int col = i % m_side;
            return (col + m_out[box_index]) / m_side;
        }

        double y(int i, int b)
        {
            int box_index = index(i, b, 0);
            int row = i / m_side;
            return (row + m_out[box_index + m_side * m_side]) / m_side;
        }

        double width(int i, int b, float anchor)
        {
            int box_index = index(i, b, 0);
            return std::exp(m_out[box_index + 2 * m_side * m_side]) * anchor / m_side;
        }

        double height(int i, int b, float anchor)
        {
            int box_index = index(i, b, 0);
            return std::exp(m_out[box_index + 3 * m_side * m_side]) * anchor / m_side;
        }

        float classConf(int i, int b, int label)
        {
            int class_index = index(i, b, m_lcoords + 1 + label);
            return m_out[class_index];
        }
    };

    class YoloParams {
    public:
        int num = 5;
        int coords = 4;
    };

    cv::Rect toBox(double x, double y, double h, double w, cv::Size in_sz)
    {
        auto h_scale = in_sz.height;
        auto w_scale = in_sz.width;
        Rect r;
        r.x = static_cast<int>((x - w / 2) * w_scale);
        r.y = static_cast<int>((y - h / 2) * h_scale);
        r.width = static_cast<int>(w * w_scale);
        r.height = static_cast<int>(h * h_scale);
        return r;
    }
} // anonymous namespace

/** YOLO kernel implementation */
GAPI_OCV_KERNEL(GOCVParseYoloWithConf, GParseYoloWithConf)
{
    static void run(const Mat & in_yolo_result,
        const Size & in_size,
        float confidence_threshold,
        float nms_threshold,
        const GYoloAnchors & anchors,
        std::vector<Rect> & out_boxes,
        std::vector<int> & out_labels,
        std::vector<float> & out_confidences)
    {
        auto dims = in_yolo_result.size;
        // We can accept several shapes in this parser:
        // If we get a rank 2 tensor, we need to make sure we can reshape it into {1, 13, 13, N*5}.
        cv::Mat yolo_result;
        if (dims.dims() == 2)
        {
            // Try to reshape
            GAPI_Assert(dims[0] == 1);
            GAPI_Assert((dims[1] / (13 * 13)) % 5 == 0);
            yolo_result = in_yolo_result.reshape(1, std::vector<int>{1, 13, 13, (dims[1] / (13 * 13))});
            dims = yolo_result.size;
        }
        else
        {
            yolo_result = in_yolo_result;
        }

        GAPI_Assert(dims.dims() == 4);
        GAPI_Assert(dims[0] == 1);
        // Accept {1,1,1,N*13*13*5} or {1,13,13,N*5}
        GAPI_Assert(((dims[1] == 1) && (dims[2] == 1) && (dims[3] % (5 * 13 * 13) == 0)) ||
            ((dims[1] == 13) && (dims[2] == 13) && (dims[3] % 5 == 0)));
        const auto num_classes = dims[3] * dims[2] * dims[1] / (5 * 13 * 13) - 5;
        GAPI_Assert(num_classes > 0);
        GAPI_Assert(0 < nms_threshold && nms_threshold <= 1);

        out_boxes.clear();
        out_labels.clear();
        out_confidences.clear();

        YoloParams params;
        constexpr auto side = 13;
        constexpr auto side_square = side * side;
        const auto output = yolo_result.ptr<float>();

        YoloParser parser(output, side, params.coords, num_classes);

        struct Detection {
            cv::Rect rect;
            float    conf;
            int      label;
        };
        std::vector<Detection> detections;

        for (int i = 0; i < side_square; i++)
        {
            for (int b = 0; b < params.num; b++)
            {
                float scale = parser.scale(i, b);
                if (scale < confidence_threshold)
                {
                    continue;
                }
                double x = parser.x(i, b);
                double y = parser.y(i, b);
                double height = parser.height(i, b, anchors[2 * b + 1]);
                double width = parser.width(i, b, anchors[2 * b]);

                for (int label = 0; label < num_classes; label++)
                {
                    float prob = scale * parser.classConf(i, b, label);
                    if (prob < confidence_threshold)
                    {
                        continue;
                    }
                    auto box = toBox(x, y, height, width, in_size);

                    detections.emplace_back(Detection{ box, prob, label });
                }
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
                        })) {
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

/** C++ wrapper for the YOLO parser */
GAPI_EXPORTS GDetectionsWithConf parseYoloWithConf(const GMat& in, const GOpaque<Size>& in_sz, float confidence_threshold = 0.5f, float nms_threshold = 0.5f, const GYoloAnchors& anchors = GParseYoloWithConf::defaultAnchors());

} // namespace streaming
} // namespace gapi
} // namespace cv
