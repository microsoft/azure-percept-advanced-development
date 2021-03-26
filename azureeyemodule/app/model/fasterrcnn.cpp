// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <thread>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "fasterrcnn.hpp"
#include "../kernels/ssd_kernels.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"


namespace model {

/**
 * The Faster RCNN network takes two inputs: the image, which will be cropped and downsampled down to 600x1024,
 * and some information about the image, shaped [1, 3], where the first item is batch size and the last item
 * is a vector of three values: height, width, scale-factor (1).
 *
 * The Faster RCNN network outputs a single output, which we will parse into boxes, labels, and confidences.
 */
G_API_NET(FasterRCNNNetwork, <cv::GMat(cv::GMat)>, "faster-rcnn-network");


FasterRCNNModel::FasterRCNNModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : ObjectDetector{ labelfpath, modelfpaths, mvcmd, videofile, resolution }
{
}

void FasterRCNNModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        this->wait_for_device();

        // Read in the labels for classification
        label::load_label_file(this->class_labels, this->labelfpath);
        this->log_parameters();

        // Build the camera pipeline with G-API
        *pipeline = this->compile_cv_graph();
        util::log_info("starting the pipeline...");
        pipeline->start();

        // Pull data through the pipeline
        bool ran_out_naturally = this->pull_data(*pipeline);

        if (!ran_out_naturally)
        {
            break;
        }
    }
}

cv::GStreamingCompiled FasterRCNNModel::compile_cv_graph() const
{
    cv::GMat in;
    cv::GMat preproc = cv::gapi::mx::preproc(in, this->resolution);
    cv::GArray<uint8_t> h264;
    cv::GOpaque<int64_t> h264_seqno;
    cv::GOpaque<int64_t> h264_ts;
    std::tie(h264, h264_seqno, h264_ts) = cv::gapi::streaming::encH264ts(preproc);

    // We have BGR output and H264 output in the same graph.
    // In this case, BGR always must be desynchronized from the main path
    // to avoid internal queue overflow (FW reports this data to us via
    // separate channels)
    // copy() is required only to maintain the graph contracts
    // (there must be an operation following desync()). No real copy happens
    cv::GMat img = cv::gapi::copy(cv::gapi::streaming::desync(preproc));

    // This branch has inference and is desynchronized to keep
    // a constant framerate for the encoded stream (above)
    cv::GMat bgr = cv::gapi::streaming::desync(preproc);

    cv::GMat nn = cv::gapi::infer<FasterRCNNNetwork>(bgr);

    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(nn);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    cv::GArray<cv::Rect> rcs;
    cv::GArray<int> ids;
    cv::GArray<float> cfs;

    // Faster RCNN can use the exact same parser as SSD
    std::tie(rcs, ids, cfs) = cv::gapi::streaming::parseSSDWithConf(nn, sz);

    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,      // main path: H264 (~constant framerate)
                                  img,                                     // desynchronized path: BGR
                                  nn_seqno, nn_ts, rcs, ids, cfs, sz));    // Inference path

    auto detector = cv::gapi::mx::Params<FasterRCNNNetwork>{this->modelfiles.at(0)};

    // Faster RCNN requires this constant input thing
    cv::Mat constant_imginfo(cv::Size{3, 1}, CV_32FC1);
    auto ptr = constant_imginfo.ptr<float>();
    ptr[0] = 600;
    ptr[1] = 1024;
    ptr[2] = 1;
    detector.constInput("image_info", constant_imginfo);
    detector.cfgInputLayers({"image_tensor"});

    auto networks = cv::gapi::networks(detector);

    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseSSDWithConf>());

    // Compile the graph in streamnig mode, set all the parameters
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline, and start processing
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

void FasterRCNNModel::log_parameters() const
{
    // Log all the stuff
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += "firmware: " + this->mvcmd + ", parser: Faster RCNN, label: " + this->labelfpath + ", classes: " + std::to_string((int)this->class_labels.size());
    util::log_info(msg);
}

} // namespace model
