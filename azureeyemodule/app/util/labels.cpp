// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <string>
#include <vector>

// Local includes
#include "helper.hpp"
#include "labels.hpp"


namespace label {

/* Colors to be used for bounding boxes, etc. */
static const std::vector<cv::Scalar> the_colors = {
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0)
};

const std::vector<cv::Scalar>& colors()
{
    return the_colors;
}

void load_label_file(std::vector<std::string> &class_labels, const std::string &labelfile)
{
    util::log_info("Loading label file " + labelfile);
    std::ifstream file(labelfile);

    if (file.is_open())
    {
        class_labels.clear();

        std::string line;
        while (getline(file, line))
        {
            // remove \r in the end of line
            if (!line.empty() && line[line.size() - 1] == '\r')
            {
                line.erase(line.size() - 1);
            }
            class_labels.push_back(line);
        }
        file.close();
    }
    else
    {
        util::log_error("Could not open file " + labelfile);
    }
}

} // namespace label
