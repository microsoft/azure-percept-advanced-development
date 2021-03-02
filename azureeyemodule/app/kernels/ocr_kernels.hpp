// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Some of this is derived from some demo code found here: https://github.com/openvinotoolkit/open_model_zoo
// And taken under Apache License 2.0
// Copyright (C) 2020 Intel Corporation
#pragma once

// Standard library includes
#include <vector>

//Third party includes
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/mx.hpp> // size()

namespace cv {
namespace gapi {
namespace streaming {

using CVGSize = cv::GOpaque<cv::Size>;
using GRRects = cv::GArray<cv::RotatedRect>;

G_API_OP(PostProcess, <GRRects(cv::GMat,cv::GMat,CVGSize,float,float)>,"sample.custom.text.post_proc")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &, const cv::GOpaqueDesc &, float, float)
    {
        return cv::empty_array_desc();
    }
};

using GMats = cv::GArray<cv::GMat>;

G_API_OP(CropLabels, <GMats(cv::GMat,GRRects,cv::Size)>, "sample.custom.text.crop")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GArrayDesc &, const cv::Size &)
    {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProcess, PostProcess)
{
    static void run (const cv::Mat &link, const cv::Mat &segm, const cv::Size &img_size, const float link_threshold, const float segm_threshold, std::vector<cv::RotatedRect> &out)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const int kMinArea = 300;
        const int kMinHeight = 10;

        const float *link_data_pointer = link.ptr<float>();
        std::vector<float> link_data(link_data_pointer, link_data_pointer + link.total());
        link_data = transpose4d(link_data, dimsToShape(link.size), {0, 2, 3, 1});
        softmax(link_data);
        link_data = sliceAndGetSecondChannel(link_data);

        std::vector<int> new_link_data_shape = {
                link.size[0],
                link.size[2],
                link.size[3],
                link.size[1]/2,
        };

        const float *cls_data_pointer = segm.ptr<float>();
        std::vector<float> cls_data(cls_data_pointer, cls_data_pointer + segm.total());
        cls_data = transpose4d(cls_data, dimsToShape(segm.size), {0, 2, 3, 1});
        softmax(cls_data);
        cls_data = sliceAndGetSecondChannel(cls_data);
        std::vector<int> new_cls_data_shape = {
                segm.size[0],
                segm.size[2],
                segm.size[3],
                segm.size[1]/2,
        };

        out = maskToBoxes(decodeImageByJoin(cls_data, new_cls_data_shape, link_data, new_link_data_shape, segm_threshold, link_threshold),
                                            static_cast<float>(kMinArea), static_cast<float>(kMinHeight), img_size);
    }

    static std::vector<std::size_t> dimsToShape(const cv::MatSize &sz)
    {
        const int n_dims = sz.dims();
        std::vector<std::size_t> result;
        result.reserve(n_dims);

        // cv::MatSize is not iterable...
        for (int i = 0; i < n_dims; i++)
        {
            result.emplace_back(static_cast<std::size_t>(sz[i]));
        }
        return result;
    }

    static void softmax(std::vector<float> &rdata)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const size_t last_dim = 2;
        for (size_t i = 0 ; i < rdata.size(); i+=last_dim)
        {
            float m = std::max(rdata[i], rdata[i+1]);
            rdata[i] = std::exp(rdata[i] - m);
            rdata[i + 1] = std::exp(rdata[i + 1] - m);
            float s = rdata[i] + rdata[i + 1];
            rdata[i] /= s;
            rdata[i + 1] /= s;
        }
    }

    static std::vector<float> transpose4d(const std::vector<float> &data, const std::vector<size_t> &shape, const std::vector<size_t> &axes)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        if (shape.size() != axes.size())
        {
            throw std::runtime_error("Shape and axes must have the same dimension.");
        }

        for (size_t a : axes)
        {
            if (a >= shape.size())
            {
                throw std::runtime_error("Axis must be less than dimension of shape.");
            }
        }

        size_t total_size = shape[0]*shape[1]*shape[2]*shape[3];

        std::vector<size_t> steps {
            shape[axes[1]]*shape[axes[2]]*shape[axes[3]],
            shape[axes[2]]*shape[axes[3]],
            shape[axes[3]],
            1
        };

        size_t source_data_idx = 0;
        std::vector<float> new_data(total_size, 0);
        std::vector<size_t> ids(shape.size());

        for (ids[0] = 0; ids[0] < shape[0]; ids[0]++)
        {
            for (ids[1] = 0; ids[1] < shape[1]; ids[1]++)
            {
                for (ids[2] = 0; ids[2] < shape[2]; ids[2]++)
                {
                    for (ids[3]= 0; ids[3] < shape[3]; ids[3]++)
                    {
                        size_t new_data_idx = ids[axes[0]]*steps[0] + ids[axes[1]]*steps[1] + ids[axes[2]]*steps[2] + ids[axes[3]]*steps[3];
                        new_data[new_data_idx] = data[source_data_idx++];
                    }
                }
            }
        }
        return new_data;
    }

    static std::vector<float> sliceAndGetSecondChannel(const std::vector<float> &data)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        std::vector<float> new_data(data.size() / 2, 0);
        for (size_t i = 0; i < data.size() / 2; i++)
        {
            new_data[i] = data[2 * i + 1];
        }
        return new_data;
    }

    static void join(const int p1, const int p2, std::unordered_map<int, int> &group_mask)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const int root1 = findRoot(p1, group_mask);
        const int root2 = findRoot(p2, group_mask);
        if (root1 != root2)
        {
            group_mask[root1] = root2;
        }
    }

    static cv::Mat decodeImageByJoin(const std::vector<float> &cls_data,
                                     const std::vector<int>   &cls_data_shape,
                                     const std::vector<float> &link_data,
                                     const std::vector<int>   &link_data_shape,
                                     float cls_conf_threshold,
                                     float link_conf_threshold)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const int h = cls_data_shape[1];
        const int w = cls_data_shape[2];

        std::vector<uchar> pixel_mask(h * w, 0);
        std::unordered_map<int, int> group_mask;
        std::vector<cv::Point> points;
        for (int i = 0; i < static_cast<int>(pixel_mask.size()); i++)
        {
            pixel_mask[i] = cls_data[i] >= cls_conf_threshold;
            if (pixel_mask[i])
            {
                points.emplace_back(i % w, i / w);
                group_mask[i] = -1;
            }
        }
        std::vector<uchar> link_mask(link_data.size(), 0);
        for (size_t i = 0; i < link_mask.size(); i++)
        {
            link_mask[i] = link_data[i] >= link_conf_threshold;
        }

        size_t neighbours = size_t(link_data_shape[3]);
        for (const auto &point : points)
        {
            size_t neighbour = 0;
            for (int ny = point.y - 1; ny <= point.y + 1; ny++)
            {
                for (int nx = point.x - 1; nx <= point.x + 1; nx++)
                {
                    if (nx == point.x && ny == point.y)
                    {
                        continue;
                    }

                    if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                    {
                        uchar pixel_value = pixel_mask[size_t(ny) * size_t(w) + size_t(nx)];
                        uchar link_value = link_mask[(size_t(point.y) * size_t(w) + size_t(point.x)) *neighbours + neighbour];
                        if (pixel_value && link_value)
                        {
                            join(point.x + point.y * w, nx + ny * w, group_mask);
                        }
                    }
                    neighbour++;
                }
            }
        }
        return get_all(points, w, h, group_mask);
    }

    static cv::Mat get_all(const std::vector<cv::Point> &points, const int w, const int h, std::unordered_map<int, int> &group_mask)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        std::unordered_map<int, int> root_map;
        cv::Mat mask(h, w, CV_32S, cv::Scalar(0));
        for (const auto &point : points)
        {
            int point_root = findRoot(point.x + point.y * w, group_mask);
            if (root_map.find(point_root) == root_map.end())
            {
                root_map.emplace(point_root, static_cast<int>(root_map.size() + 1));
            }
            mask.at<int>(point.x + point.y * w) = root_map[point_root];
        }
        return mask;
    }

    static int findRoot(const int point, std::unordered_map<int, int> &group_mask)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        int root = point;
        bool update_parent = false;
        while (group_mask.at(root) != -1)
        {
            root = group_mask.at(root);
            update_parent = true;
        }

        if (update_parent)
        {
            group_mask[point] = root;
        }
        return root;
    }

    static std::vector<cv::RotatedRect> maskToBoxes(const cv::Mat &mask, const float min_area, const float min_height, const cv::Size &image_size)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        std::vector<cv::RotatedRect> bboxes;
        double min_val = 0.;
        double max_val = 0.;
        cv::minMaxLoc(mask, &min_val, &max_val);
        int max_bbox_idx = static_cast<int>(max_val);
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, image_size, 0, 0, cv::INTER_NEAREST);

        for (int i = 1; i <= max_bbox_idx; i++)
        {
            cv::Mat bbox_mask = resized_mask == i;
            std::vector<std::vector<cv::Point>> contours;

            cv::findContours(bbox_mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty())
            {
                continue;
            }

            cv::RotatedRect r = cv::minAreaRect(contours[0]);
            if (std::min(r.size.width, r.size.height) < min_height)
            {
                continue;
            }

            if (r.size.area() < min_area)
            {
                continue;
            }
            bboxes.emplace_back(r);
        }
        return bboxes;
    }
}; // GAPI_OCV_KERNEL(PostProcess)

GAPI_OCV_KERNEL(OCVCropLabels, CropLabels)
{
    static void run(const cv::Mat &image, const std::vector<cv::RotatedRect> &detections, const cv::Size &outSize, std::vector<cv::Mat> &out)
    {
        out.clear();
        out.reserve(detections.size());
        cv::Mat crop(outSize, CV_8UC3, cv::Scalar(0));
        cv::Mat gray(outSize, CV_8UC1, cv::Scalar(0));
        std::vector<int> blob_shape = {1,1,outSize.height,outSize.width};

        for (auto &&rr : detections)
        {
            std::vector<cv::Point2f> points(4);
            rr.points(points.data());

            const auto top_left_point_idx = topLeftPointIdx(points);
            cv::Point2f point0 = points[static_cast<size_t>(top_left_point_idx)];
            cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
            cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

            std::vector<cv::Point2f> from{point0, point1, point2};
            std::vector<cv::Point2f> to {
                cv::Point2f(0.0f, 0.0f),
                cv::Point2f(static_cast<float>(outSize.width-1), 0.0f),
                cv::Point2f(static_cast<float>(outSize.width-1),
                            static_cast<float>(outSize.height-1))
            };
            cv::Mat M = cv::getAffineTransform(from, to);
            cv::warpAffine(image, crop, M, outSize);
            cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

            cv::Mat blob = gray;
            gray.convertTo(blob, CV_32F);
            out.push_back(blob.reshape(1, blob_shape)); // pass as 1,1,H,W instead of H,W
        }
    }

    static int topLeftPointIdx(const std::vector<cv::Point2f> &points)
    {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        int most_left_idx = -1;
        int almost_most_left_idx = -1;

        for (size_t i = 0; i < points.size() ; i++)
        {
            if (most_left.x > points[i].x)
            {
                if (most_left.x < std::numeric_limits<float>::max())
                {
                    almost_most_left = most_left;
                    almost_most_left_idx = most_left_idx;
                }
                most_left = points[i];
                most_left_idx = static_cast<int>(i);
            }

            if (almost_most_left.x > points[i].x && points[i] != most_left)
            {
                almost_most_left = points[i];
                almost_most_left_idx = static_cast<int>(i);
            }
        }

        if (almost_most_left.y < most_left.y)
        {
            most_left = almost_most_left;
            most_left_idx = almost_most_left_idx;
        }
        return most_left_idx;
    }
}; // GAPI_OCV_KERNEL(CropLabels)

} // namespace streaming
} // namespace gapi
} // namespace cv
