// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "ssd_kernels.hpp"

namespace cv {
namespace gapi {
namespace custom {

GDetectionsWithConf parse_ssd_with_confidences(const GMat& in, const GOpaque<Size> &in_sz, float confidence_threshold, int filter_label)
{
    return GParseSSDWithConf::on(in, in_sz, confidence_threshold, filter_label);
}

} // namespace custom
} // namespace gapi
} // namespace cv
