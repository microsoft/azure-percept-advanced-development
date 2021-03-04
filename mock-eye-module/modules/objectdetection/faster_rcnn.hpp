/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This module contains the Faster RCNN ResNet parser.
 * It is the easiest to understand, as it does everything in a single function.
 * Once you understand this example, you should probably consider structuring your code
 * more like it is in object_detectors.*.
 */
#pragma once

// Standard library includes
#include <string>
#include <vector>

// Our includes
#include "../device.hpp"
#include "../parser.hpp"

namespace detection {
namespace rcnn {

/**
 * Compiles the GAPI graph for an object detection model and runs the application. This method never returns.
 *
 * @param video_fpath: If given, we run the model on the given movie. If empty, we use the webcam.
 * @param modelfpath: The path to the model's .xml file.
 * @param weightsfpath: The path to the model's .bin file.
 * @param device: What device we should run on.
 * @param show: If true, we display the results.
 * @param labels: The labels this model was built to detect.
 */
void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels);

} // namespace rcnn
} // namespace detection
