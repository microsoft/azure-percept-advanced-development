// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <iomanip>
#include <string>
#include <vector>

// Third party library includes
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

// Our includes
#include "../../kernels/ssd_kernels.hpp"
#include "../../kernels/yolo_kernels.hpp"
#include "../../kernels/utils.hpp"
#include "../device.hpp"
#include "object_detectors.hpp"

namespace detection {

/** Declare an ObjectDetector network type that we can use for inference in GAPI. It takes one matrix and outputs one matrix. */
G_API_NET(ObjectDetector, <cv::GMat(cv::GMat)>, "com.microsoft.azure.object-detector");

// This is the function that we use to build a G-API graph for the SSD network.
// Feel free to copy it.
static cv::GStreamingCompiled build_inference_graph_ssd(const std::string &modelfile, const std::string &weightsfile, const device::Device &device)
{
    // The input node. We will fill this in with OpenCV Mat objects as we get them from our video source.
    cv::GMat in;

    // Apply the network to the frame to produce a single output image
    cv::GMat nn = cv::gapi::infer<ObjectDetector>(in);

    // Get the size of the input image. We'll need this for later.
    cv::GOpaque<cv::Size> sz = cv::gapi::custom::size(in);

    // Copy the raw input. We'll use this for overlaying our detections on top of later.
    cv::GMat bgr = cv::gapi::copy(in);

    // Here we call our own custom G-API op on the neural network inferences and the size of the input image
    // to get the bounding boxes, the predicted class labels, and our confidences.
    // See the kernels/ssd.hpp and kernels/ssd.cpp for the implementation of parse_ssd_with_confidences.
    cv::GArray<cv::Rect> boxes;
    cv::GArray<int> ids;
    cv::GArray<float> confidences;
    std::tie(boxes, ids, confidences) = cv::gapi::custom::parse_ssd_with_confidences(nn, sz);

    // Set up the actual kernels (the implementations of the parser ops)
    auto kernels = cv::gapi::combine(cv::gapi::kernels<cv::gapi::custom::GOCVParseSSDWithConf>(),
                                     cv::gapi::kernels<cv::gapi::custom::GOCVSize>(),
                                     cv::gapi::kernels<cv::gapi::custom::GOCVSizeR>());

    // Bundle the GAPI params (this is also where we instantiate the backend - in this application,
    // the backend is Inference Engine; in the Percept DK, it is a custom backend for the device).
    auto net = cv::gapi::ie::Params<ObjectDetector>{ modelfile, weightsfile, device::device_to_string(device) };

    // Set up the inputs and outputs of the graph.
    auto comp = cv::GComputation(cv::GIn(in), cv::GOut(bgr, nn, boxes, ids, confidences));

    // Now compile the graph.
    auto compiled_args = cv::compile_args(cv::gapi::networks(net), kernels);
    auto cs = comp.compileStreaming(std::move(compiled_args));

    return cs;
}

// This is the function that we use to build a G-API graph for the YOLO networks.
// Feel free to copy it.
static cv::GStreamingCompiled build_inference_graph_yolo(const std::string &modelfile, const std::string &weightsfile, const device::Device &device)
{
    // The input node. We will fill this in with OpenCV Mat objects as we get them from our video source.
    cv::GMat in;

    // Apply the network to the frame to produce a single output image
    cv::GMat nn = cv::gapi::infer<ObjectDetector>(in);

    // Get the size of the input image. We'll need this for later.
    cv::GOpaque<cv::Size> sz = cv::gapi::custom::size(in);

    // Copy the raw input. We'll use this for overlaying our detections on top of later.
    cv::GMat bgr = cv::gapi::copy(in);

    // Here we call our own custom G-API op on the neural network inferences and the size of the input image
    // to get the bounding boxes, the predicted class labels, and our confidences.
    // See the kernels/yolo.hpp and kernels/yolo.cpp files for the implementation of parse_yolo_with_confidences.
    cv::GArray<cv::Rect> boxes;
    cv::GArray<int> ids;
    cv::GArray<float> confidences;
    std::tie(boxes, ids, confidences) = cv::gapi::custom::parse_yolo_with_confidences(nn, sz);

    // Set up the actual kernels (the implementations of the parser ops)
    auto kernels = cv::gapi::combine(cv::gapi::kernels<cv::gapi::custom::GOCVParseYoloWithConf>(),
                                     cv::gapi::kernels<cv::gapi::custom::GOCVSize>(),
                                     cv::gapi::kernels<cv::gapi::custom::GOCVSizeR>());

    // Bundle the GAPI params (this is also where we instantiate the backend - in this application,
    // the backend is Inference Engine; in the Percept DK, it is a custom backend for the device).
    auto net = cv::gapi::ie::Params<ObjectDetector>{ modelfile, weightsfile, device::device_to_string(device) };

    // Set up the inputs and outputs of the graph.
    auto comp = cv::GComputation(cv::GIn(in), cv::GOut(bgr, nn, boxes, ids, confidences));

    // Now compile the graph.
    auto compiled_args = cv::compile_args(cv::gapi::networks(net), kernels);
    auto cs = comp.compileStreaming(std::move(compiled_args));

    return cs;
}

/**
 * Builds the inference graph, but does not yet run it.
 *
 * @param video_in Path to the input video if we are using an input video file instead of web cam.
 * @param parser The Parser we will be using to interpret the results of the inference graph.
 * @param modelfile Path to the model's IR (topology).
 * @param weightsfile Path to the model's weights.
 * @param device Device we are using.
 * @returns The compiled graph.
 */
static cv::GStreamingCompiled build_inference_graph(const std::string &video_in, const parser::Parser &parser,
                                                    const std::string &modelfile, const std::string &weightsfile, const device::Device &device)
{
    cv::GStreamingCompiled cs;
    switch (parser)
    {
        case parser::Parser::SSD100:  // Fall-through
        case parser::Parser::SSD200:
            cs = build_inference_graph_ssd(modelfile, weightsfile, device);
            break;
        case parser::Parser::YOLO:
            cs = build_inference_graph_yolo(modelfile, weightsfile, device);
            break;
        default:
            std::cerr << "Have not yet implemented this Parser's logic." << std::endl;
            exit(__LINE__);
    }

    if (video_in.empty())
    {
        // Specify the web cam as the input to the pipeline.
        cs.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(-1));
    }
    else
    {
        // Specify the user-supplied video file as the input to the pipeline.
        cs.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(video_in));
    }

    return cs;
}

/**
 * Look up the label index in the class list.
 *
 * @param index The class label as integer.
 * @param classes The list of class labels.
 * @returns The class label corresponding to the integer.
 */
static std::string get_label(size_t index, const std::vector<std::string> &classes)
{
    if (index < classes.size())
    {
        return classes[index];
    }
    else
    {
        return std::to_string(index);
    }
}

/**
 * Returns a string representation of the given value down to a particular precision.
 *
 * @param f The floating point value to convert to a string.
 * @param precision The amount of precision to use.
 * @returns A string representation of `f`
 */
static std::string to_string_with_precision(float f, int precision)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << f;
    return ss.str();
}

/**
 * Preview the given image.
 *
 * @param bgr The BGR image
 * @param boxes The bounding boxes to display
 * @param labels The labels in the same order as the bounding boxes
 * @param confidences The confidences of each bounding box (in the same order)
 * @param classes The list of class label names
 */
static void preview(const cv::Mat &bgr, const std::vector<cv::Rect> &boxes, const std::vector<int> &labels, const std::vector<float> &confidences, const std::vector<std::string> &classes)
{
    // colors to be used for bounding boxes
    static std::vector<cv::Scalar> colors = {
            cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
            cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85),
            cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
            cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
            cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
            cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0)
    };

    for (std::size_t i = 0; i < boxes.size(); i++)
    {
        // Deal with annoying int->size_t conversion
        size_t label_i;
        if (labels[i] < 0)
        {
            continue;
        }
        else
        {
            label_i = labels[i];
        }

        // color of a label
        size_t index = label_i % colors.size();

        cv::rectangle(bgr, boxes[i], colors.at(index), 2);
        cv::putText(bgr,
            get_label(label_i, classes) + ": " + to_string_with_precision(confidences[i], 2),
            boxes[i].tl() + cv::Point(3, 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(colors.at(index)),
            2);
    }
}

/**
 * Pulls the data from the graph pipeline forever. This is where we run the G-API graph we compiled.
 *
 * @param cs The compiled streaming graph
 * @param show If true, we visualize the display (requires a GUI)
 * @param classes The list of class label names
 */
void pull_data_from_pipeline(cv::GStreamingCompiled &cs, bool show, const std::vector<std::string> &classes)
{
    // Set up all the output nodes. This will be filled in by the cs.pull method.
    cv::Mat out_bgr;
    cv::Mat out_nn;
    std::vector<cv::Rect> out_boxes;
    std::vector<int> out_labels;
    std::vector<float> out_confidences;

    // This while loop is where we pull the data from the camera or file and run the network on it.
    while (cs.pull(cv::gout(out_bgr, out_nn, out_boxes, out_labels, out_confidences)))
    {
        // After cs.pull() is called, we have filled in all of our output nodes.
        // There are ways to fill in nodes asynchronously, but in this application,
        // we do not bother. In the Azure Percept DK azureeyemodule, we do.

        // Visualize the output of the network.
        preview(out_bgr, out_boxes, out_labels, out_confidences, classes);

        // Preview if humans are watching us run this on a GUI
        if (show)
        {
            cv::imshow("preview", out_bgr);
            cv::waitKey(1);
        }
    }
}

// The entry point for the object detector network parsers. We build a G-API graph, start it, then run it.
void compile_and_run(const std::string &video_fpath, const parser::Parser &parser, const std::string &modelfpath,
                     const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels)
{
    auto cs = build_inference_graph(video_fpath, parser, modelfpath, weightsfpath, device);
    cs.start();
    pull_data_from_pipeline(cs, show, labels);
}

} // namespace detection
