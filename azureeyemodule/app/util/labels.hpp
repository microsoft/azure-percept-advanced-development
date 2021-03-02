// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>
#include <vector>

// Third party includes
#include <opencv2/core.hpp>


namespace label {

/** Retrieve the color vector. */
const std::vector<cv::Scalar>& colors();

/** Load in label file to fill the class_labels. */
void load_label_file(std::vector<std::string> &class_labels, const std::string &labelfile);

} // namespace label
