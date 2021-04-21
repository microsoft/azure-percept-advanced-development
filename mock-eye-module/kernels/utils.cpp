// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * Utility kernels and ops
 */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include "utils.hpp"


namespace cv {
namespace gapi {
namespace custom {

GOpaque<Size> size(const GMat& in) {
    return GSize::on(in);
}

GOpaque<Size> size(const GOpaque<Rect>& in) {
    return GSizeR::on(in);
}

} // custom namespace
} // gapi namespace
} // cv namespace
