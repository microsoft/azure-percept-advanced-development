# OpenVINO Model Zoo Tutorial

**Please note!** The experiences in this repository should be considered to be in **preview/beta**.
Significant portions of these experiences are subject to change without warning. **No part of this code should be considered stable**.

**Note** This tutorial will not work until we incorporate the latest OpenVINO version into the azureeyemodule!

This tutorial shows you how to start from an OpenVINO model zoo neural network that has already been trained to do something
fairly general, and then port that model to the Azure Percept DK.

## Select a Model

If you have a need for a visual AI pipeline that can be solved by a model found in a zoo, such as general object detection,
person detection, etc., you can take a look at [the OpenVINO model zoo](https://docs.openvinotoolkit.org/2021.1/omz_models_intel_index.html).

For the purposes of this tutorial, we will be working with a simple [semantic segmentation model](https://docs.openvinotoolkit.org/2021.1/omz_models_intel_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html).

You can see from that page some basic information about this model:

* It was trained on 20 classes (all of which seem to be outside/on the road).
* Its input is an image of size 2048x1024 (scrolling down the page to the bottom, you can see it really requires
  an input of shape [batch size, n-channels, 1024, 2048]).
* It has 6.686 million parameters.

If you will recall from the PyTorch from Scratch tutorial, we don't know how large is too large for a network,
because we have a large memory buffer that must be shared by several things, including the camera frame buffer,
the compiled G-API graph, the network(s), etc. But a rule of thumb is to try to keep your network under 50 MB,
and definitely under 100 MB.

Note that this device does not allow for int8 quantization schemes, though we can use FP16. Let's assume
we go with FP32 just to see though, since this network is 6.686x10^6 parameters,
multiply this by four bytes per parameter, and we get 26,744,000 bytes, or about 26 MB. This network is easily small
enough to fit in our device, even if we use FP32. FP16 will halve that size.

## Example Code

While we've tried to make the experience of bringing your own AI model pipeline to the Azure Percept DK as easy as possible,
it is the nature of these models that some code will have to be written. In particular, the post-processing of the neural network
is something that you will have to take care of, because we can't know ahead of time what you will want to do with a network's output.
This will be a lot easier if you have some example code to go off of.

So, let's find some sample code for this model. Oh look, [here it is](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/segmentation_demo_async/main.cpp).
Note the license is Apache 2.0.

Let's try creating a super simple OpenCV application that tests this model out. First, let's just run the sample as is, then we'll strip it
down to just the part(s) that we care about.

To try running this sample unfortunately requires installing OpenVINO, which is outside the scope of this tutorial, as it is not
really necessary to port a model to the device. If you end up using a lot of models from the Intel Open Model Zoo, you should
definitely do this, but otherwise just take it from me that this sample does in fact work.

Let's instead integrate this example into our mock-eye-module application.

## Mock Eye Module

At the root of this repository is a [mock-eye-module](../../mock-eye-module/README.md) folder, which is useful for porting models from
PC to the Azure Percept DK. You could instead port the model directly into the azureeyemodule application and deploy it onto the device,
but, being an embedded device, the Azure Percept DK is not a great tool for going through debugging cycles.

So let's port this main.cpp file into the Mock Eye Module and make sure it works there before we port it all the way over to the device.

If you look at the mock-eye-module directory, you can see it contains a C++ application that can be compiled and run using
Docker. Let's build it and run it before making any changes to it, just to make sure it works.

If you haven't done the PyTorch from Scratch tutorial, then you will need to satisfy some prerequisites first.
See the [relavent instructions](../pytorch-from-scratch-tutorial/pytorch-from-scratch-tutorial.md#Mock-Eye-Module)
before continuing here.

## Semantic Segmentation

Now that we've got a sandbox to test our model in, let's work on porting the semantic segmentation model over to it.

The first thing to do is to use the OpenVINO Workbench again. This time, instead of downloading SSD, search for
"semantic", and that should be good enough to find "semantic-segmentation-adas-0001". Import it, download it, and extract it
into its two files. Put the files somewhere where you won't lose them.

Now let's go through what you will need to do to add support for this model to the mock-eye-module. Remember the point of adding
support for this model to the mock-eye-module is that doing so will put us about halfway towards our real goal of porting
this model to the Percept application instead. This sandbox application will allow us to run GDB and to feed a movie file
as an input to the application, while also being a much smaller application that is easier to reason about.

Here are the steps that are needed to add support to the mock-eye-module:

1. Add a new "parser" variant to the enum in mock-eye-module/modules/parser.[c/h]pp,
   and don't forget to update the `look_up_parser()` function there so the command line can accept your parser as an argument.
1. Add a folder under `modules` called `segmentation`, and then update the CMakeLists.txt file to include the new folder.
1. Put all of our runtime logic in the `modules/segmentation` folder, which will include compiling a G-API graph,
   passing the graph our custom kernels (which we will make in this tutorial), and then running the graph, collecting
   the graph's outputs, and interpreting them.
1. Implement whatever custom kernels we need for our G-API graph.

Once we have completed these steps, porting the resulting logic to the Percept DK should be pretty simple, meanwhile, we'll complete all
of these steps on our host PC, which should make development quite a bit more comfortable.

Let's go through these steps one at a time.

### Parser Enum

In order to integrate our new model (and its post-processing logic - i.e., "parser") into the mock app, we need to tell the command line
arguments and the main function that we have a new AI model that we can accept.

Let's start by updating the enum and look-up function in `mock-eye-module/modules/parser.hpp` and `mock-eye-module/modules/parser.cpp`:

First, here's the .hpp file:

```C++
// The contents of this enum may have changed by the time you read this,
// because maybe I forgot to update this documentation. But either way,
// find this enum and update it to include "UNET_SEM_SEG" or whatever you want to call it.
enum class Parser {
    FASTER_RCNN,
    OPENPOSE,
    SSD100,
    SSD200,
    // Here's the new item
    UNET_SEM_SEG,
    YOLO
};
```

Next, here's the .cpp file:

```C++
Parser look_up_parser(const std::string &parser_str)
{
    if (parser_str == "openpose")
    {
        return Parser::OPENPOSE;
    }
    else if (parser_str == "ssd100")
    {
        return Parser::SSD100;
    }
    else if (parser_str == "ssd200")
    {
        return Parser::SSD200;
    }
    else if (parser_str == "yolo")
    {
        return Parser::YOLO;
    }
    else if (parser_str == "faster-rcnn")
    {
        return Parser::FASTER_RCNN;
    }
    else if (parser_str == "unet-seg") ///////// This is the new one
    {                                  /////////
        return Parser::UNET_SEM_SEG;   /////////
    }                                  /////////
    else
    {
        std::cerr << "Given " << parser_str << " for --parser, but we do not support it." << std::endl;
        exit(-1);
    }
}
```

Now update main.cpp:

```C++
// Make sure to include this
#include "modules/segmentation/unet_semseg.hpp"

// .... other code

/** Arguments for this program (short-arg long-arg | default-value | help message) */
static const std::string keys =
"{ h help    |        | Print this message }"
"{ d device  | CPU    | Device to run inference on. Options: CPU, GPU, NCS2 }"
"{ p parser  | ssd100 | Parser kind required for input model. Possible values: ssd100, ssd200, yolo, openpose, faster-rcnn, unet-seg }" // Update the help message
"{ w weights |        | Weights file }"
"{ x xml     |        | Network XML file }"
"{ labels    |        | Path to the labels file }"
"{ show      | false  | Show output BGR image. Requires graphical environment }"
"{ video_in  |        | If given, we use this file as input instead of the camera }";

// .... clip some more code

// Here we are in main():
    std::vector<std::string> classes;
    switch (parser)
    {
        case parser::Parser::OPENPOSE:
            pose::compile_and_run(video_in, xml, weights, dev, show);
            break;
        case parser::Parser::SSD100:  // Fall-through
        case parser::Parser::SSD200:  // Fall-through
        case parser::Parser::YOLO:
            classes = load_label(labelfile);
            detection::compile_and_run(video_in, parser, xml, weights, dev, show, classes);
            break;
        case parser::Parser::FASTER_RCNN:
            classes = load_label(labelfile);
            detection::rcnn::compile_and_run(video_in, xml, weights, dev, show, classes);
            break;
        case parser::Parser::UNET_SEM_SEG:                                        // NEW CODE
            classes = load_label(labelfile);                                      // NEW CODE
            semseg::compile_and_run(video_in, xml, weights, dev, show, classes);  // NEW CODE
            break;                                                                // NEW CODE
        default:
            std::cerr << "Programmer error: Please implement the appropriate logic for this Parser." << std::endl;
            exit(__LINE__);
    }
```

So now we've updated all the logic we need to route the application's flow to the right place if the user executes
this application with the `--parser unet-seg` argument.

Of course, this won't compile yet, since we don't have a `semseg` at all, let alone a `compile_and_run` function in it.
So let's code up that function now.

### Modules/Segmentation

Create a folder where we will put all of our semantic segmentation code: `mkdir modules/segmentation`.

Now let's create the header file, which will be entirely boilerplate:

```C++
// Put this in a file called mock-eye-module/modules/segmentation/unet_semseg.hpp

/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#pragma once

// Standard library includes
#include <string>
#include <vector>

// Our includes
#include "../device.hpp"
#include "../parser.hpp"

namespace semseg {

/**
 * Compiles the GAPI graph for a semantic segmentation model (U-Net, specifically) and runs the application. This method never returns.
 *
 * @param video_fpath: If given, we run the model on the given movie. If empty, we use the webcam.
 * @param modelfpath: The path to the model's .xml file.
 * @param weightsfpath: The path to the model's .bin file.
 * @param device: What device we should run on.
 * @param show: If true, we display the results.
 * @param labels: The labels this model was built to detect.
 */
void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels);

} // namespace semseg
```

How did I know that that's the code I should put in the header? By looking at `modules/objectdetection/faster_rcnn.hpp`.

Each of the arguments to the single function that we need to implement is explained in the header file, but
to be more verbose:

* `video_fpath`: Must be a valid path to a video file. Note that on Windows, the webcam won't work - only video files are supported.
* `modelfpath`: The path to the model's .xml file. Remember that each model is in the OpenVINO IR format, and therefore is composed
  of a topology (.xml) file and a weights (.bin) file.
* `weightsfpath`: The path to the model's .bin file.
* `device`: We haven't talked about devices. If you are curious, you can check in the device module, but the gist of it is that since the Inference Engine
  OpenCV back end we use in this application supports GPUs, Myriad X VPUs, and CPUs, I figured we could just support all of them. Unfortunately for Windows
  users, only CPU is supported.
* `show`: We don't need to show the GUI, but it is cool (and helpful for debugging). You could certainly get a way with just using
  `std::cout` messages.
* `labels`: Our U-Net model is trained to do semantic segmentation on particular items. If we don't give this over to the function,
  we'll make sure that the function just displays numbers instead of labels, so it is technically optional. Nonetheless, we'll pass something in either way,
  and if the function can't find the given file (perhaps because it is just an empty string, and not a file path at all), then we'll ignore this arg
  and output numbers instead of letters.

Let's add the .cpp file now.

```C++
// Put this in a file called mock-eye-module/modules/segmentation/unet_semseg.cpp
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <iomanip>
#include <random>
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
#include "../../kernels/unet_semseg_kernels.hpp"
#include "../device.hpp"
#include "../parser.hpp"
#include "unet_semseg.hpp"

namespace semseg {

// This macro is used to tell G-API what types this network is going to take in and output.
// In our case, we are going to take in a single image (represented as a CV Mat, in G-API called a GMat)
// and output a tensor of shape {Batch size (which will be 1), 1, 1024 pixels high, 2048 pixels wide},
// which we will again represent as a GMat.
//
// The tag at the end can be anything you want.
//
// In case you are wondering, it is <output(input)>
G_API_NET(SemanticSegmentationUNet, <cv::GMat(cv::GMat)>, "com.microsoft.u-net-semseg-network");

// I know I wanted to only have one function in this file, but it is really helpful to just have
// this single other function for visualization purposes. Don't worry about it, it's taken
// pretty much directly from the semantic segmentation demo I pointed you to on GitHub earlier
// and just overlays colors on top of an input image and returns it as an output image.
cv::Mat apply_color_map(const cv::Mat &input)
{
    // We creat a static colors "array". It is really a shape {256, 1, 3} tensor.
    static cv::Mat colors;

    // Create a random number generator to create our colors randomly. Use the same seed each time.
    static std::mt19937 rng;
    rng.seed(12345);
    static std::uniform_int_distribution<int> distr(0, 255);

    // If this is the first time we call this function, we will fill in the colors array
    if (colors.empty())
    {
        colors = cv::Mat(256, 1, CV_8UC3);
        for (size_t i = 0; i < (std::size_t)colors.cols; i++)
        {
            colors.at<cv::Vec3b>(i, 0) = cv::Vec3b(distr(rng), distr(rng), distr(rng));
        }
    }

    // Converting class to color
    cv::Mat out;
    cv::applyColorMap(input, out, colors);
    return out;
}

// This is the only function we expose in the header file, so it is technically the only function we **need**.
// We will only use a single function and put all of our stuff inside it for the sake of the tutorial,
// but if you were doing this for real, you would probably want to break stuff out into separate functions.
void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels)
{
    // Create the network itself. Here we are using the cv::gapi::ie namespace, which stands for Inference Engine.
    // On the device, we have a custom back end, namespaced as cv::gapi::mx instead.
    auto network = cv::gapi::ie::Params<SemanticSegmentationUNet>{ modelfpath, weightsfpath, device::device_to_string(device) };

    // Graph construction //////////////////////////////////////////////////////

    // Construct the input node. We will fill this in with OpenCV Mat objects as we get them from the video source.
    cv::GMat in;

    // Apply the network to the frame to produce a single output image, as specified by the parameters we used
    // in the G_API_NET macro.
    auto nn = cv::gapi::infer<SemanticSegmentationUNet>(in);

    // Get the size of the input image. We'll need this for later.
    auto sz = cv::gapi::custom::size(in);

    // Here we call our own custom G-API op on the neural network inferences and the size of the input image
    // to interpret the output of the neural network. G-API will resize the input image for us, and it will
    // run the network for us on whichever device we passed into the Params constructor,
    // but it obviously doesn't know what to do with the output.
    //
    // So we will write up some custom code to interpret the output and we will incorporate
    // that output into the graph here.
    //
    // As this is a semantic segmentation network, we'd like this graph op to take in the output of our
    // neural network (which should be an image of size {1024 rows, 2048 cols}, where each pixel
    // is a 2D coordinate with a value that corresponds to the class it belongs to)
    // and then color it based on the pixel values. Lastly, we'll resize it back down to the same size as the
    // input image. We'll also output the IDs of the objects we see, so that we can report them to the
    // console later.
    //
    // We'll talk in detail about this later, but obviously, we'll need to write this function ourselves.
    cv::GArray<int> ids;
    cv::GMat colored_output;
    std::tie(colored_output, ids) = cv::gapi::custom::parse_unet_for_semseg(nn, sz);

    // Since we want to be able to overlay the semantic segmentation mask on top of
    // the original image, we will also need to feed the original image out of the graph.
    auto raw_input = cv::gapi::copy(in);

    // These are all the output nodes for the graph.
    auto graph_outs = cv::GOut(colored_output, ids, raw_input);

    // Graph compilation ///////////////////////////////////////////////////////

    // The G-API graph makes use of a bunch of default ops, but also some custom ones.
    // Because G-API separates the concept of interface from implementation,
    // we have to specify the interface (the op, which we did in the graph above)
    // and the particular implementation of the ops that we use (the kernels, which
    // we specify here).
    //
    // We will need to provide these kernels. The size ones have already been done for you,
    // but we will need to write up the GOCVParseUnetForSemSeg kernel ourselves.
    auto kernels = cv::gapi::kernels<cv::gapi::custom::GOCVParseUnetForSemSeg,
                                     cv::gapi::custom::GOCVSize,
                                     cv::gapi::custom::GOCVSizeR>();

    // Set up the inputs and outpus of the graph.
    auto comp = cv::GComputation(cv::GIn(in), std::move(graph_outs));

    // Now compile the graph into a pipeline object that we will use as
    // an abstract black box that takes in images and outputs images (because this is semantic segmentation).
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

    // Set up all the output nodes.
    // Each data container needs to match the type of the G-API item that we used as a stand in
    // in the GOut call above. And the order of these data containers needs to match the order
    // that we specified in the GOut call above.
    //
    // Also, it is possible to make the G-API graph asynchronous so that each
    // item is delivered as quickly as it can. In fact, we do this in the Azure Percept azureeyemodule's application
    // so that we can output the raw RTSP stream at however fast it comes in, regardless of how fast the neural
    // network is running.
    //
    // In synchronous mode (the default), no item is output until all the output nodes have
    // something to output.
    //
    // We'll just use synchronous mode here.
    cv::Mat out_color_mask;
    std::vector<int> out_ids;
    cv::Mat out_raw_mat;
    auto pipeline_outputs = cv::gout(out_color_mask, out_ids, out_raw_mat);

    // Pull the information through the compiled graph, filling our output nodes at each iteration.
    while (pipeline.pull(std::move(pipeline_outputs)))
    {
        // Log all the objects that we see. In a more useful application,
        // we might want to specify the coordinates and maybe the confidences as well,
        // but for this example, this will suffice.
        for (const auto &id : out_ids)
        {
            const auto label = ((id > 0) && ((size_t)id < labels.size())) ? labels.at(id) : "Unknown";
            std::cout << "detected: " << id << ", label: " << label << std::endl;
        }

        // Display the semantic segmentation color mark-ups.
        if (show)
        {
            cv::imshow("Out", (out_raw_mat / 2) + (out_color_mask / 2)); // Taken from the segmentation demo
            cv::waitKey(1);
        }
    }
}

} // namespace semseg

```

### Custom Kernel

Now, if you are like me and have done everything in your power to make sure that your editor can find all the header files and therefore
has intellisense (or your equivalent) enabled, you will see that we have two red squiggles to take care of:

* `std::tie(colored_output, ids) = cv::gapi::custom::parse_unet_for_semseg(nn, sz);`
* `auto kernels = cv::gapi::kernels<cv::gapi::custom::GOCVParseUnetForSemSeg,`

We need to provide implementations for `parse_unet_for_semseg` and `GOCVParseUnetForSemSeg`.

The first is a C++ function that will wrap our G-API op. The second is the kernel implementation of the G-API op that we are wrapping.
Which means we really have three things we need to write:

* A G-API op (essentially a function signature wrapped in some boilerplate)
* A C++ wrapper for the op (pretty much just boilerplate)
* A G-API kernel for the op (here's where all the good stuff will be)

Let's knock out the op first, since it is required for the other two.

```C++
// Put this in mock-eye-module/kernels/unet_semseg_kernels.hpp
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard libary includes
#include <tuple>
#include <vector>

// Third party includes
#include <opencv2/opencv.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>


namespace cv {
namespace gapi {
namespace custom {

/** Type alias for detections along with an image. */
using GSegmentationAndClasses = std::tuple<GMat, GArray<int>>;

// Op for parsing the output of the U-Net semantic segmentation network.
// This macro creates a G-API op. Remember, an op is the interface for a G-API "function",
// which can be written into a G-API graph.
//
// Since it is just the interface and not the implementation, we merely describe the input and the
// output of the operation here, while leaving the actual implementation up to the kernel we will
// write down below.
//
// For a little explanation: the macro takes three arguments:
// G_API_OP(op name, op function signature, op tag)
//
// The op name can be whatever you want. We have adopted the Intel convention of labeling ops
// GWhatever using camel case. Kernels are also camel case and follow the format GOCVWhatever.
// C++ wrapper functions use snake_case. Feel free to do whatever you want though.
//
// The function signature looks like this <output args(input args)>. Because C++ does not have native
// support for multiple return values (like Python), we need to wrap the multiple return values into
// a std::tuple, and we went ahead and aliased it into GSegmentationAndClasses to be more readable.
// A note about the types: in order to insert an op into a G-API graph, you need to make sure the types
// are mapped from what you actually want into the G-API type system.
// G-API types are simple: they have support for primatives, for cv::Mat objects as GMat objects,
// vectors as GArray objects, and everything else as GOpaque<type>.
//
// The tag can be whatever you want, and frankly, I'm not even sure what it's used for...
G_API_OP(GParseUnetForSemSeg, <GSegmentationAndClasses(GMat, GOpaque<Size>)>, "org.microsoft.gparseunetsemseg")
{
    // This boilerplate is required within curly braces following the macro.
    // You declare a static function called outMeta, which must take in WhateverDesc
    // versions of the types and output the same thing.
    //
    // Notice that we have mapped our inputs from GMat -> GMatDesc and GOpaque<Size> -> GOpaqueDesc
    // (and added const and reference declaration syntax to them).
    // We've also changed GMat -> GMatDesc and GArray -> GArrayDesc.
    static std::tuple<GMatDesc, GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&)
    {
        // Just return an empty desc for each of these things.
        return std::make_tuple(empty_gmat_desc(), empty_array_desc());
    }
};

// We'll put more code down here before we close out the namespace
```

We'll need a little more boilerplate. Let's create a C++ function that wraps our op.

```C++
// Put this in mock-eye-module/kernels/unet_semseg_kernels.cpp
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "unet_semseg_kernels.hpp"

namespace cv {
namespace gapi {
namespace custom {

// This is pretty much pure boilerplate and is not strictly necessary.
// We just wrap the invocation of our op with a traditional C++ function
// so that we can just call parse_unet_for_semseg() rather than use
// the stranger looking GParseUnetForSemSeg::on() syntax.
// Really, that's all.
GSegmentationAndClasses parse_unet_for_semseg(const GMat &in, const GOpaque<Size> &in_sz)
{
    return GParseUnetForSemSeg::on(in, in_sz);
}

} // namespace custom
} // namespace gapi
} // namespace cv
```

With that boilerplate taken care of, let's create an actual implementation of the op: the kernel.

```C++
// Put this at the bottom of mock-eye-module/kernels/unet_semseg_kernels.cpp

// This is the kernel declaration for the op.
// A single op can have several kernel implementations. That's kind of the whole point.
// The idea behind G-API is twofold: one is to make a declarative style computer vision
// pipeline syntax, and the other is to separate the declaration of the computer vision pipeline
// from its implementation. The reason for this is because you may want to have the same code
// that runs on a VPU, GPU, or CPU. If you are using G-API for it, all you would need to do
// is implement a GOCVParseUnetForSemSegCPU, GOCVParseUnetForSemSegGPU, and a GOCVParseUnetForSemSegVPU
// function. All the other code would remain the same.
//
// In our case, we are going to do everything in this function the CPU, since there's not really
// any acceleration needed for this, and because our VPU on the device is occupied running
// the neural network and doing a few other things.
//
// So we will just create a single kernel, and it will run on the CPU.
GAPI_OCV_KERNEL(GOCVParseUnetForSemSeg, GParseUnetForSemSeg)
{
    // We need a static void run function.
    // It needs to take in the inputs as references and output the return values as references.
    //
    // So, since our op is of type <GSegmentationAndClasses(GMat, GOpaque<Size>)>
    // which decays to <std::tuple<GMat, GArray<int>>(GMat, GOpaque<Size>)>
    // and since the kernel function needs to run good old fashioned C++ code (not G-API code),
    // we need to map this type to:
    //
    // <std::tuple<cv::Mat, std::vector<int>>(cv::Mat, cv::Size)>
    //
    // but because we need to map our return value to a list of output references,
    // the actual signature of this function is:
    // <void(const Mat&, const Size&, Mat&, std::vector<int>&)>
    static void run(const Mat &in_img, const Size &in_sz, Mat &out_img, std::vector<int> &out_classes)
    {
        // Here's where we will implement all the logic for post-processing our neural network
        // Since this is a super simple semantic segmentation network, all the network does
        // is output a color mask. Since we will want to display this color mask overlayed on top of
        // the input image, let's resize the color mask here to the right size.
        //
        // First, remember that we aren't getting an image per se, instead, we are getting a tensor.
        // {Batch size (which will be 1), 1, 1024 pixels high, 2048 pixels wide}
        //
        // So, we need to reshape into {1024, 2048, 1} first, and then resize into whatever size we have.
        CV_Assert(in_img.size.dims() == 4);
        auto reshaped_img = in_img.reshape(1, {in_img.size[2], in_img.size[3]});
        resize(reshaped_img, out_img, in_sz);

        // Now go through all the pixel values in the input image and add them to the output classes.
        // There may be some super fast way to do this. But who cares? Let's just do a simple brute-force
        // naive algorithm: run over each pixel in the image and add it to a set. Then convert the set to
        // a vector.
        std::unordered_set<int> classes;
        auto it = in_img.begin<uchar>();
        auto end = in_img.end<uchar>();
        for (; it != end; ++it)
        {
            classes.insert(*it);
        }

        for (auto id : classes)
        {
            out_classes.push_back(id);
        }
    }
};

// Don't forget to declare the C++ wrapper function in our header!
GAPI_EXPORTS GSegmentationAndClasses parse_unet_for_semseg(const GMat &in, const GOpaque<Size> &in_sz);

} // namespace custom
} // namespace gapi
} // namespace cv
```

Phew! That was a lot. It is important to recognize something: almost all of what we've done so far is just boilerplate.
Every model that you port will follow this exact same series of steps, and will look remarkably similar to what we
have so far done.

The only custom pieces of code are the G-API graph itself and any kernel implementations.
In this example, that means the G-API code found in `compile_and_run` in `unet_semseg.cpp`
and the kernel in `unet_semseg_kernels.hpp`.

## Label file

Since our semantic segmentation network is going to be classifying objects it sees and coloring
them, it should also write what they are as well. In order to do that, it will need a label file.
I didn't create this network, so I don't know what the labels are which correspond to the outputs,
but OpenVINO does: put the following into a labels.txt file.

```
road
sidewalk
building
wall
fence
pole
traffic light
traffic sign
vegetation
terrain
sky
person
rider
car
truck
bus
train
motorcycle
bicycle
ego-vehicle
```

What's an "ego-vehicle"? I didn't know. I'm still not sure I know, but I looked it up and it seems to be
a driverless car or something... Whatever.

## Compilation and Debugging

Now let's try to compile it. Because I am so nice, I made sure that the compiler errors were all fixed before
posting this tutorial. But for pedagogical purposes, I've left logical errors; let's walk through the debug cycles together.

First, let's compile and run. You will need the semantic-segmentation-adas-0001 model in IR format. I already told you to
download it and put it somewhere. You can get it from the OpenVINO Workbench, in case you didn't already do that.
You will also need a movie file (or if you are running on Linux, you can use the webcam on your computer instead).
Since this model was trained on traffic scenes, I recommend using a video that has cars and whatnot in it.
Intel has a GitHub that hosts some small video files for their demos, so you could [check that out](https://github.com/intel-iot-devkit/sample-videos),
in particular, car-detection.mp4 and person-bicycle-car-detection.mp4 are both fine. Or use your own.

I'll use person-bicycle-car-detection.mp4 for this tutorial, but I'll try to make sure it works for car-detection.mp4 as well.

Once you have both of those things, you can try compiling and running the mock-eye-module with:

```ps1
# On Windows
./scripts/compile_and_test.ps1 -ipaddr <your-ip-address> -xml <path to the XML file> -parser unet-seg -video <path to the video> -labels <path to labels.txt>
```

or

```bash
# On Linux
./scripts/compile_and_test.sh --video=<path to the video file> --weights=<path to the .bin> --xml=<path to the .xml> --labels=<path to labels.txt>
```

It should compile and run the program inside a Docker container.

First it should spout a whole bunch of GStreamer warnings, since we are running in a Docker container,
we haven't installed most of the GStreamer libraries, so OpenCV is complaining that GStreamer is missing
plugins. But we don't care.

Then it should die with a `cv::Exception` and an error message something along the lines of

```
what():  OpenCV(4.5.0-openvino) ../opencv/modules/gapi/src/backends/ie/giebackend.cpp:101: error: (-215:Assertion failed) false && "Unsupported data type" in function 'toCV'
```

Great. So... somewhere in the Inference Engine back end of the G-API library, it is hitting an assert and dying. At least it was nice enough to tell
us that specifically, it died because it has an "Unsupported data type" that it is probably trying to convert or something. Hmm.

Unfortunately, the official OpenVINO Docker images do not contain GDB, so we have to make our own Docker image for debugging.

Before I forget, remove the tmp folder that the script created. Do that after running the script each time, or add it to the script itself.
I have left the tmp folder in case for some reason I want to get at or inspect the artifacts that end up in it.

Now, let's make the new Docker image. It is very simple, just cd into the mock-eye-module directory and run

```bash
docker build . -t mock-eye-module-debug
```

It will take a few minutes to build the new Dockerfile, and when you are done, you can run `docker image list`
and see that you have made a `mock-eye-module-debug` image. It is important that you name it this, because the
compile_and_run scripts expect this name.

Now that we've built that Docker image, we can use GDB:

```ps1
# On Windows
./scripts/compile_and_test.ps1 -ipaddr <your-ip-address> -xml <path to the XML file> -parser unet-seg -video <path to the video> -labels <path to labels.txt> -debugmode
```

or

```bash
# On Linux
./scripts/compile_and_test.sh --video=<path to the video file> --weights=<path to the .bin> --xml=<path to the .xml> --labels=<path to labels.txt> --debug
```

This should build the application in the new Docker container and drop you into GDB with the right arguments.

Let's just run the program to the point where it crashes and then examine the backtrace:

```
(gdb) run
```

This command will run the program and you will see all the same stuff as before (this time augmented by a bunch of GDB messages as well).
Now you should see something like this:

```
[New Thread 0x7fffcdffb700 (LWP 310)]
[ WARN:0] global ../opencv/modules/videoio/src/cap_gstreamer.cpp (898) open OpenCV | GStreamer warning: unable to query duration of stream
[ WARN:0] global ../opencv/modules/videoio/src/cap_gstreamer.cpp (935) open OpenCV | GStreamer warning: Cannot query video position: status=1, value=0, duration=-1
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.5.0-openvino) ../opencv/modules/gapi/src/backends/ie/giebackend.cpp:101: error: (-215:Assertion failed) false && "Unsupported data type" in function 'toCV'


Thread 1 "mock_eye_app" received signal SIGABRT, Aborted.
__GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:51
51      ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
(gdb)
```

Don't worry about the "../sysdeps/unix/sysv/linux/raise.c: No such file or directory" - it just
means that we are running GDB on a lightweight Docker container that doesn't have all the
debugging goodies installed and configured.

Hitting `bt` followed by `ENTER` will get you something like this:

```
(gdb) bt
#0  0x00007ffff334ffb7 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:51
#1  0x00007ffff3351921 in __GI_abort () at abort.c:79
#2  0x00007ffff3d44957 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6
#3  0x00007ffff3d4aae6 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6
#4  0x00007ffff3d4ab21 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6
#5  0x00007ffff3d4ad54 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6
#6  0x00007ffff42a01c6 in cv::error(cv::Exception const&) () at /opt/intel/openvino/opencv/lib/libopencv_core.so.4.5
#7  0x00007ffff42a1232 in cv::error(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, char const*, char const*, int) ()
    at /opt/intel/openvino/opencv/lib/libopencv_core.so.4.5
#8  0x00007ffff7b2ad34 in cv::gimpl::ie::Infer::outMeta(ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5
#9  0x00007ffff7b1fd6c in std::_Function_handler<std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > (ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&), std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > (*)(ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> >
> const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&)>::_M_invoke(std::_Any_data const&, ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > >
const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5
#10 0x00007ffff7a290b2 in cv::gimpl::passes::inferMeta(ade::passes::PassContext&, bool) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5
---Type <return> to continue, or q <return> to quit---
#11 0x00007ffff7a0b3c9 in cv::gimpl::GCompiler::runMetaPasses(ade::Graph&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > const&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5
#12 0x00007ffff7a5de68 in cv::gimpl::GStreamingExecutor::setSource(std::vector<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef>, std::allocator<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef> > >&&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5
#13 0x00007ffff7a15683 in cv::GStreamingCompiled::Priv::setSource(std::vector<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef>, std::allocator<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef> > >&&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5
#14 0x000055555564bd4b in semseg::compile_and_run(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>,
std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, device::Device const&, bool, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) (video_fpath="/home/openvino/tmp/movie.mp4", modelfpath="/home/openvino/tmp/model.xml", weightsfpath="/home/openvino/tmp/model.bin", device=@0x7fffffffe030: device::Device::CPU, show=true, labels=std::vector of length 0, capacity 0)
    at /home/openvino/tmp/modules/segmentation/unet_semseg.cpp:142
#15 0x00005555555fb284 in main(int, char**) (argc=7, argv=0x7fffffffe268) at /home/openvino/tmp/main.cpp:132
```

This is pretty hard to read, so I've spaced each function call out for you and relisted below:

```
(gdb) bt
#0  0x00007ffff334ffb7 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:51

#1  0x00007ffff3351921 in __GI_abort () at abort.c:79

#2  0x00007ffff3d44957 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6

#3  0x00007ffff3d4aae6 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6

#4  0x00007ffff3d4ab21 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6

#5  0x00007ffff3d4ad54 in  () at /usr/lib/x86_64-linux-gnu/libstdc++.so.6

#6  0x00007ffff42a01c6 in cv::error(cv::Exception const&) () at /opt/intel/openvino/opencv/lib/libopencv_core.so.4.5

#7  0x00007ffff42a1232 in cv::error(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, char const*, char const*, int) ()
    at /opt/intel/openvino/opencv/lib/libopencv_core.so.4.5

#8  0x00007ffff7b2ad34 in cv::gimpl::ie::Infer::outMeta(ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5

#9  0x00007ffff7b1fd6c in std::_Function_handler<std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > (ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&), std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > (*)(ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> >
> const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&)>::_M_invoke(std::_Any_data const&, ade::Graph const&, ade::Handle<ade::Node> const&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > >
const&, std::vector<cv::GArg, std::allocator<cv::GArg> > const&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5

#10 0x00007ffff7a290b2 in cv::gimpl::passes::inferMeta(ade::passes::PassContext&, bool) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5

#11 0x00007ffff7a0b3c9 in cv::gimpl::GCompiler::runMetaPasses(ade::Graph&, std::vector<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc>, std::allocator<cv::util::variant<cv::util::monostate, cv::GMatDesc, cv::GScalarDesc, cv::GArrayDesc, cv::GOpaqueDesc> > > const&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5

#12 0x00007ffff7a5de68 in cv::gimpl::GStreamingExecutor::setSource(std::vector<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef>, std::allocator<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef> > >&&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5

#13 0x00007ffff7a15683 in cv::GStreamingCompiled::Priv::setSource(std::vector<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef>, std::allocator<cv::util::variant<cv::UMat, std::shared_ptr<cv::gapi::wip::IStreamSource>, cv::Mat, cv::Scalar_<double>, cv::detail::VectorRef, cv::detail::OpaqueRef> > >&&) () at /opt/intel/openvino/opencv/lib/libopencv_gapi.so.4.5

#14 0x000055555564bd4b in semseg::compile_and_run(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>,
std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, device::Device const&, bool, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) (video_fpath="/home/openvino/tmp/movie.mp4", modelfpath="/home/openvino/tmp/model.xml", weightsfpath="/home/openvino/tmp/model.bin", device=@0x7fffffffe030: device::Device::CPU, show=true, labels=std::vector of length 0, capacity 0)
    at /home/openvino/tmp/modules/segmentation/unet_semseg.cpp:142

#15 0x00005555555fb284 in main(int, char**) (argc=7, argv=0x7fffffffe268) at /home/openvino/tmp/main.cpp:132
```

Reading this from bottom to top, we get something like the following call chain:

```
main -> semseg::compile_and_run -> setSource -> setSource -> runMetaPasses -> inferMeta -> anonymous-function-handler -> outMeta -> error tracing stuff
```

So it looks like we call `main()`, which calls `compile_and_run()`, which then calls `setSource()`. All of that is in our code,
and is easy to follow. Calling `setSource()` however, invokes the G-API's just-in-time compiler, which then attempts to compile our graph.
It looks like that's where things go arry. First, it goes through an internal `setSource()` function, then off to `runMetaPasses()`,
then `inferMeta()`, followed by some anonymous function, and finally into some internal `outMeta()`, where it complains about some type not being allowed.

Looking at the source code for this file reveals that the function we are dying in is trying a switch statement between
two values: `CV_8U` and `CV_32F`, which looks like 8-bit unsigned vs 32-bit floating point. We haven't specified any types
like that.

Let's double check all of the types that are flowing through the G-API graph.

I made this pretty picture to show what's going on with the G-API graph right now:

![G-API SemSeg Graph](imgs/g-api-graph.png "G-API Semantic Segmentation Graph")

You can see that it isn't super complicated. The input image is fed into three different nodes, `infer()`, `size()`, and `copy()`.

I have double checked the image type from the video file, it is `CV_8U3`, which makes sense,
since it is an 8-bit 3-channel input image. That should be fine.

The result of `copy()` should be an identical copy, so I don't think that's the problem.

The result of `size()` is a `cv::Size` object, so that's probably not it either. Also, I have used `copy()`
and `size()` in several other graphs, and never had this problem, so I don't think those are the culprits.

So I think we can narrow this down to a handful of possibilities:

1. `infer()` has a problem with the input image.
1. `infer()` is outputting a type that is incorrect.
1. `parse_unet_for_semseg()` has a problem with `nn` or with `sz`.
1. `parse_unet_for_semseg()` is outputting something with an incorrect type.

Let's try removing both of these problematic ops from the graph and see if everything works.
I'll leave it as an exercise for you to do that, but it does indeed work just fine. So that
confirms that most of this all works. The problem is somewhere in the G-API graph having to do with
the `infer()` or the `parse_unet_for_semseg()` functions.

So let's try adding the `infer()` function back in, and just pass its output straight out of the graph,
rather than feeding it into the parser function. Again, I'll leave this as an exercise for the reader.
It results in getting the same error. So the error is in the infer function. It seems that
one of the precisions in the neural network is not supported. In particular, for input and outputs
of your neural network, you must make sure that the precisions are 8-bit unsigned integer, or 32-bit floating point.
