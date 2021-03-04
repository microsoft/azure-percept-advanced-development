// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Derived from an example from Intel:
// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Standard library includes
#include <iomanip>
#include <string>
#include <vector>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

// Our includes
#include "../../kernels/utils.hpp"
#include "../../kernels/fasterrcnn_kernels.hpp"
#include "../../kernels/ssd_kernels.hpp"
#include "../device.hpp"
#include "../parser.hpp"
#include "faster_rcnn.hpp"

namespace detection {
namespace rcnn {

G_API_NET(FasterRCNNNet, <cv::GMat(cv::GMat)>, "com.microsoft.faster-rcnn-network");

void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels)
{
    // Create the network itself.
    auto network = cv::gapi::ie::Params<FasterRCNNNet>{ modelfpath, weightsfpath, device::device_to_string(device) };

    // OpenVINO's implementation of Faster RCNN has two inputs, though one of them is constant.
    // So we need to configure the input layers here.
    cv::Mat image_info(cv::Size{3, 1}, CV_32FC1);
    auto ptr = image_info.ptr<float>();
    ptr[0] = 600;
    ptr[1] = 1024;
    ptr[2] = 1;
    network.constInput("image_info", image_info);
    network.cfgInputLayers({"image_tensor"});

    // Graph construction //////////////////////////////////////////////////////

    // Construct the input node. We will fill this in with OpenCV Mat objects as we get them from the video source.
    cv::GMat in;

    // Apply the network to the frame to produce a single output image.
    auto nn = cv::gapi::infer<FasterRCNNNet>(in);

    // Get the size of the input image. We'll need this for later.
    auto sz = cv::gapi::custom::size(in);

    // Copy the raw input image. We'll use this for overlaying our detections on top of later.
    auto out = cv::gapi::copy(in);

    // Here we call our own custom G-API op on the neural network inferences and the size of the input image
    // to get the bounding boxes and the predicted labels.
    // Note that we are actually using the SSD parsing function for this, as it turns out that
    // the two networks have compatible output formats in OpenVINO model zoo.
    cv::GArray<cv::Rect> boxes;
    cv::GArray<int> ids;
    cv::GArray<float> confidences;
    std::tie(boxes, ids, confidences) = cv::gapi::custom::parse_ssd_with_confidences(nn, sz);

    // These are all the output nodes for the graph.
    auto graph_outs = cv::GOut(boxes, ids, confidences, out);

    // Graph compilation ///////////////////////////////////////////////////////

    // Set up the actual kernels (the implementations of the parser ops)
    auto kernels = cv::gapi::kernels<cv::gapi::custom::GOCVParseSSDWithConf,
                                     cv::gapi::custom::GOCVSize,
                                     cv::gapi::custom::GOCVSizeR,
                                     cv::gapi::custom::OCVBBoxes>();

    // Set up the inputs and outpus of the graph.
    auto comp = cv::GComputation(cv::GIn(in), std::move(graph_outs));

    // Now compile the graph.
    auto compiled_args = cv::compile_args(kernels, cv::gapi::networks(network));
    auto pipeline = comp.compileStreaming(std::move(compiled_args));

    // Graph execution /////////////////////////////////////////////////////////

    // Select a video source - either the webcam or an input file.
    if (!video_fpath.empty())
    {
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(video_fpath)));
    }
    else
    {
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(-1)));
    }

    // Now start the pipeline
    pipeline.start();

    // Set up all the output nodes
    std::vector<cv::Rect> out_boxes;
    std::vector<int> out_labels;
    std::vector<float> out_confidences;
    cv::Mat out_mat;
    auto pipeline_outputs = cv::gout(out_boxes, out_labels, out_confidences, out_mat);

    // Pull the information through the compiled graph, filling each output node at each iteration of the loop.
    while (pipeline.pull(std::move(pipeline_outputs)))
    {
        // Log what we see.
        for (std::size_t i = 0; i < out_boxes.size(); i++)
        {
            std::cout << "detected: " << out_boxes[i]
                      << ", label: "  << out_labels[i]
                      << ", conf: "   << std::fixed << std::setprecision(2) << out_confidences[i]
                      << std::endl;
        }

        // Display what we see as bounding boxes with labels.
        if (show)
        {
            for (std::size_t i = 0; i < out_boxes.size(); i++)
            {
                cv::rectangle(out_mat, out_boxes[i], cv::Scalar(0, 255, 0), 2);
                cv::putText(out_mat,
                            "label " + std::to_string(out_labels[i]),
                            out_boxes[i].tl() + cv::Point(3, 20),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.7,
                            cv::Scalar(0, 255, 0),
                            2);
            }
            cv::imshow("Out", out_mat);
            cv::waitKey(1);
        }
    }
}

} // namespace rcnn
} // namespace detection
