// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard libary includes
#include <tuple>
#include <vector>

// Third party includes
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>


namespace cv {
namespace gapi {
namespace custom {

/** Type alias for SSD bounding boxes, IDs, and confidences */
using GDetectionsWithConf = std::tuple<GArray<Rect>, GArray<int>, GArray<float>>;

/** Op for parsing the SSD Outputs */
G_API_OP(GParseSSDWithConf, <GDetectionsWithConf(GMat, GOpaque<Size>, float, int)>, "org.opencv.dnn.parseSSDWithConf")
{
    static std::tuple<GArrayDesc, GArrayDesc, GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&, float, int)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc());
    }
};

/** Kernel implementation of the SSD parsing */
GAPI_OCV_KERNEL(GOCVParseSSDWithConf, GParseSSDWithConf)
{
    static void run(const Mat & in_ssd_result,
        const Size & in_size,
        float confidence_threshold,
        int filter_label,
        std::vector<Rect> & out_boxes,
        std::vector<int> & out_labels,
        std::vector<float> & out_confidences) {
        const auto& in_ssd_dims = in_ssd_result.size;
        GAPI_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE = in_ssd_dims[3];
        GAPI_Assert(OBJECT_SIZE == 7); // fixed SSD object size

        out_boxes.clear();
        out_labels.clear();
        out_confidences.clear();

        const auto items = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++)
        {
            const auto it = items + i * OBJECT_SIZE;
            float image_id = it[0];
            float label = it[1];
            float confidence = it[2];
            float rc_left = it[3];
            float rc_top = it[4];
            float rc_right = it[5];
            float rc_bottom = it[6];

            if (image_id < 0.f)
            {
                break;    // marks end-of-detections
            }

            if (confidence < confidence_threshold)
            {
                continue; // skip objects with low confidence
            }

            if (filter_label != -1 && static_cast<int>(label) != filter_label)
            {
                continue; // filter out object classes if filter is specified
            }

            const Rect surface({ 0,0 }, in_size);

            Rect rc;  // map relative coordinates to the original image scale
            rc.x = static_cast<int>(rc_left * in_size.width);
            rc.y = static_cast<int>(rc_top * in_size.height);
            rc.width = static_cast<int>(rc_right * in_size.width) - rc.x;
            rc.height = static_cast<int>(rc_bottom * in_size.height) - rc.y;
            out_boxes.emplace_back(rc & surface);
            out_labels.emplace_back(label);
            out_confidences.emplace_back(confidence);
        }
    }
};

/** C++ wrapper function for parsing SSD. */
GAPI_EXPORTS GDetectionsWithConf parse_ssd_with_confidences(const GMat &in, const GOpaque<Size>& in_sz, float confidence_threshold = 0.5f, int filter_label = -1);

} // namespace custom
} // namespace gapi
} // namespace cv
