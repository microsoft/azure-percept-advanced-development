// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "ssd_kernels.hpp"

namespace cv {
namespace gapi {
namespace streaming {

GDetectionsWithConf parseSSDWithConf(const GMat& in, const GOpaque<Size> &in_sz, float confidence_threshold, int filter_label)
{
    return GParseSSDWithConf::on(in, in_sz, confidence_threshold, filter_label);
}

} // namespace streaming
} // namespace gapi
} // namespace cv
