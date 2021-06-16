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
#include <opencv2/gapi/render.hpp>

// Local includes
#include "ssd.hpp"
#include "../kernels/ssd_kernels.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace custom {

using GDetections = cv::GArray<cv::Rect>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(BBoxes, <GPrims(GDetections)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // Converts the rectangles into G-API's rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }
    }
};

} // namespace custom

namespace model {

/** An SSD network takes a single input and outputs a single output (which we will parse into boxes, labels, and confidences) */
G_API_NET(SSDNetwork, <cv::GMat(cv::GMat)>, "ssd-network");

bool check_file_exist(std::string filepath)
{
   std::ifstream ifile;
   ifile.open(filepath);
   if(ifile) {
      return true;
   } else {
      return false;
   }
}

SSDModel::SSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string inputsource, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : ObjectDetector{ labelfpath, modelfpaths, mvcmd, videofile, resolution }
{
    this->inputsource = inputsource;
}

void SSDModel::load_default()
{
    // int ret = util::run_command("rm -rf /app/model && mkdir /app/model");
    // if (ret != 0)
    // {
    //     util::log_error("rm && mkdir failed with " + ret);
    // }

    this->modelfiles = {"/app/data/ssd_mobilenet_v2_coco.blob"};
    this->labelfpath = "/app/data/labels.txt";
}

void SSDModel::run(cv::GStreamingCompiled *pipeline)
{
    while (true)
    {
        // Wait for the VPU to come up.
        this->wait_for_device();

        // Read in the labels for classification
        label::load_label_file(this->class_labels, this->labelfpath);

        // Log some metadata.
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

cv::GStreamingCompiled SSDModel::compile_cv_graph() const
{
    // The input node of the G-API pipeline. This will be filled in, one frame at time.
    cv::GMat in;

    // We have a custom preprocessing node for the Myriad X-attached camera.
    //cv::GMat preproc = cv::gapi::mx::preproc(in, this->resolution);
    cv::GMat preproc;

    // This path is the H.264 path. It gets our frames one at a time from
    // the camera and encodes them into H.264.
    cv::GArray<uint8_t> h264;
    cv::GOpaque<int64_t> h264_seqno;
    cv::GOpaque<int64_t> h264_ts;
    cv::GMat bgr;
    cv::GMat img;
    auto graph_outs = cv::GOut();
    //cv::GOpaque<int64_t> nn_ts;

    if (this->inputsource == "uvc") {
        bgr = in;
        //nn_ts = cv::gapi::streaming::timestamp(bgr);
        util::log_info("input is uvc");
    }
    else if (this->inputsource.empty()) {
        preproc = cv::gapi::mx::preproc(in, this->resolution);
        std::tie(h264, h264_seqno, h264_ts) = cv::gapi::streaming::encH264ts(preproc);
        // We have BGR output and H264 output in the same graph.
        // In this case, BGR always must be desynchronized from the main path
        // to avoid internal queue overflow (FW reports this data to us via
        // separate channels)
        // copy() is required only to maintain the graph contracts
        // (there must be an operation following desync()). No real copy happens
        img = cv::gapi::copy(cv::gapi::streaming::desync(preproc));

        // This branch has inference and is desynchronized to keep
        // a constant framerate for the encoded stream (above)
        bgr = cv::gapi::streaming::desync(preproc);
    }

    // Here's where we actually run our neural network. It runs on the VPU.
    cv::GMat nn = cv::gapi::infer<SSDNetwork>(bgr);

    // Get some useful metadata.
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn);
    //cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);
    cv::GOpaque<int64_t> ts    = cv::gapi::streaming::timestamp(nn);

    // Here's where we post-process our network's outputs into bounding boxes, IDs, and confidences.
    cv::GArray<cv::Rect> rcs;
    cv::GArray<int> ids;
    cv::GArray<float> cfs;
    cv::GArray<cv::Rect> objs;
    cv::GArray<int> tags;
    //auto graph_outs = cv::GOut(objs, tags, nn_seqno, ts);
    //auto graph_outs = cv::GOut(objs, tags);
    std::tie(objs, tags) = cv::gapi::streaming::parseSSD(nn, cv::gapi::streaming::size(bgr));
    //auto rendered = cv::gapi::wip::draw::render3ch(bgr, custom::BBoxes::on(objs));
    //graph_outs += cv::GOut(rendered);
    auto graph = cv::GComputation(cv::GIn(in), cv::GOut(objs, tags, nn_seqno, ts));
    util::log_info("after computation");

    // Pass the actual neural network blob file into the graph. We assume we have a modelfiles of at least length 1.
    CV_Assert(this->modelfiles.size() >= 1);
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<SSDNetwork>{this->modelfiles.at(0)});
    util::log_info("after networks");
    // Here we wrap up all the kernels (the implementations of the G-API ops) that we need for our graph.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseSSDWithConf>());
    util::log_info("after combine");
    // Compile the graph in streamnig mode; set all the parameters; feed the firmware file into the VPU.
    auto pipeline = graph.compileStreaming(cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));
    util::log_info("********devin log" + this->inputsource);

    // Specify the Percept DK's camera as the input to the pipeline.
    if (this->inputsource.empty()) {
        // Specify the Azure Percept's Camera as the input to the pipeline, and start processing
        pipeline.setSource<cv::gapi::mx::Camera>();
    } else {
        if (check_file_exist(this->inputsource)) {
            util::log_info("********devin log: file exists" + this->inputsource);
            pipeline.setSource<cv::gapi::wip::GCaptureSource>(this->inputsource);
        }
        else if (this->inputsource == "uvc") {
            util::log_info("*******devin log: uvc");
            pipeline.setSource<cv::gapi::wip::GCaptureSource>(0); // default: video0
        }
        else 
            util::log_error("********devin log: file not existed" + this->inputsource);
    }


    return pipeline;
}

void SSDModel::log_parameters() const
{
    // Log all the stuff
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += ", firmware: " + this->mvcmd + ", parser: SSD, label: " + this->labelfpath + ", classes: " + std::to_string((int)this->class_labels.size());
    util::log_info(msg);
}

} // namespace model
