/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This module is responsible for handling the data collection loop,
 * as much as possible. Some of its responsibility must lie with the
 * neural model, since that's where the data comes from, but to
 * the largest extent possible, we delegate the duties of the retrain
 * loop to this module.
 */
#pragma once

// Third party includes
#include <opencv2/core.hpp>


namespace loop {

/** PThread function target for running the data uploading loop. */
void *export_data(void *);

/**
 * Write the given data to a file in the right location.
 *
 * `filename` should be the name, not the path, and should include the extension.
 * Additionally, `filename` MUST be of the form <confidence>-<whatever-you-want>.<extension>.
 */
void write_data_file(const std::string &filename, const cv::Mat &bgr);

} // namespace loop
