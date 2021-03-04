/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * Use this application as a test bed for adding new AI models to the Azure Percept DK azureeyemodule.
 * This application is small and easy to extend, so can be used for debugging and proving that a model
 * works before porting it to the Azure Percept DK.
 */
// Standard Library includes
#include <fstream>
#include <iostream>
#include <signal.h>
#include <string>
#include <vector>

// Local includes
#include "kernels/ssd_kernels.hpp"
#include "kernels/utils.hpp"
#include "kernels/yolo_kernels.hpp"
#include "modules/device.hpp"
#include "modules/objectdetection/faster_rcnn.hpp"
#include "modules/objectdetection/object_detectors.hpp"
#include "modules/parser.hpp"
#include "modules/openpose/pose_estimators.hpp"

/** Arguments for this program (short-arg long-arg | default-value | help message) */
static const std::string keys =
"{ h help    |        | Print this message }"
"{ d device  | CPU    | Device to run inference on. Options: CPU, GPU, NCS2 }"
"{ p parser  | ssd100 | Parser kind required for input model. Possible values: ssd100, ssd200, yolo, openpose, faster-rcnn }"
"{ w weights |        | Weights file }"
"{ x xml     |        | Network XML file }"
"{ labels    |        | Path to the labels file }"
"{ show      | false  | Show output BGR image. Requires graphical environment }"
"{ video_in  |        | If given, we use this file as input instead of the camera }";

/**
 * This is just a helper function for loading a label file.
 *
 * @param labelfile The path to the label file. Should be a file that contains one label per line.
 * @returns The class list from the file.
 */
static std::vector<std::string> load_label(const std::string &labelfile)
{
    std::vector<std::string> classes;
    std::ifstream file(labelfile);

    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            // remove \r in the end of line
            if (!line.empty() && line[line.size() - 1] == '\r')
            {
                line.erase(line.size() - 1);
            }
            classes.push_back(line);
        }

        file.close();
    }
    else
    {
        std::cerr << "Cannot open labelfile " << labelfile << std::endl;
        std::cerr << "Labels will not be available." << std::endl;
    }

    return classes;
}

/** If we receive a SIGINT, we print a message and try to exit cleanly. */
static void interrupt(int)
{
    std::cout << "Received interrupt signal." << std::endl;
    exit(0);
}

/**
 * Here's the main function.
 * To add a new AI model, all you need to do is add a corresponding "parser",
 * which is the code that is used to load the neural network and post-process its outputs.
 *
 * 1. Start by adding an enum variant to the parser enum in modules/parser.[c/h]pp.
 * 2. Then add a new module into modules/. If you are adding an object detector, you can make use
 *    of some of the code in modules/object_detectors.[c/h]pp.
 */
int main(int argc, char* argv[])
{
    // Parse the command line args
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cmd.printMessage();
        return 0;
    }

    // Assign application parameters
    const auto dev = device::look_up_device(cmd.get<std::string>("device"));
    const auto parser = parser::look_up_parser(cmd.get<std::string>("parser"));
    const auto weights = cmd.get<std::string>("weights");
    const auto xml = cmd.get<std::string>("xml");
    const auto labelfile = cmd.get<std::string>("labels");
    const auto show = cmd.get<bool>("show");
    const auto video_in = cmd.get<std::string>("video_in");

    // Set up SIGINT handler - we peform cleanup and exit the application
    signal(SIGINT, interrupt);

    // Choose the appropriate path based on the parser we've chosen (which indicates which model we are running)
    // If you are extending this application, add a new case with your parser here. Take a look at one of the other
    // compile_and_run() functions to use as an example.
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
        default:
            std::cerr << "Programmer error: Please implement the appropriate logic for this Parser." << std::endl;
            exit(__LINE__);
    }

    return 0;
}
