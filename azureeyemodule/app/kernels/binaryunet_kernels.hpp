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

G_API_OP(PostProcBinaryUnet, <cv::GMat(cv::GMat)>, "custom.unet_postproc_1channel")
{
  static cv::GMatDesc outMeta(const cv::GMatDesc &)
  {
    // This function is required for G-API engine to figure out
    // what the output format is, given the input parameters.
    return cv::GMatDesc(CV_32F, 1, {408, 308});
  }
};

/**
 * Kernel for the above op.
 *
 * We receive a 1x1x408x308 tensor and need to convert it to a 408x308 cv::Mat.
 */
GAPI_OCV_KERNEL(GOCVPostProcBinaryUnet, PostProcBinaryUnet)
{
  static void run(const cv::Mat &in_mask, cv::Mat &out_mask)
  {
    const auto &in_mask_dims = in_mask.size;

    assert(in_mask_dims.dims() == 4U);

    //size_t channels = in_mask_dims[1];
    size_t height = in_mask_dims[2];
    size_t width = in_mask_dims[3];

    const float *data = in_mask.ptr<float>();
    // row-major layout of Mat object
    size_t step_h = in_mask.step[2] / sizeof(float);
    size_t step_w = in_mask.step[3] / sizeof(float);

    //populate the output
    for (size_t row = 0; row < height; row++)
    {
      for (size_t col = 0; col < width; col++)
      {
        float val = *(data + step_h * row + step_w * col);
        out_mask.at<float>(row, col) = val;
      }
    }
  }
};
} // namespace streaming
} // namesace gapi
} // namesace cv
