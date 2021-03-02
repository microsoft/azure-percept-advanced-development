// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Derived from some demo code found here: https://github.com/openvinotoolkit/open_model_zoo
// And taken under Apache License 2.0
// Copyright (C) 2020 Intel Corporation
#pragma once

// Standard library includes
#include <vector>

// Third-party includes
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/mx.hpp> // size()


namespace ocr
{

    template<typename Iter> void softmax_and_choose(Iter begin, Iter end, int *argmax, float *prob);

    template<typename Iter> std::vector<float> softmax(Iter begin, Iter end);

    struct BeamElement
    {
        /** The sequence of chars that will be a result of the beam element */
        std::vector<int> sentence;

        /** The probability that the last char in CTC sequence for the beam element is the special blank char */
        float prob_blank;

        /** The probability that the last char in CTC sequence for the beam element is NOT the special blank char */
        float prob_not_blank;

        /** The probability of the beam element. */
        float prob() const
        {
            return prob_blank + prob_not_blank;
        }
    };

    std::string CTCGreedyDecoder(const float *data, const std::size_t sz, const std::string &alphabet, const char pad_symbol, double *conf);

    std::string CTCBeamSearchDecoder(const float *data, const std::size_t sz, const std::string &alphabet, double *conf, int bandwidth);

    struct Decoded
    {
        std::string text; double conf;
    };

    struct TextDecoder
    {
        int ctc_beam_dec_bw;
        std::string symbol_set;
        char pad_symbol;

        Decoded decode(const cv::Mat& text) const
        {
            const auto &blob = text;
            const float *data = blob.ptr<float>();
            const auto sz = blob.total();
            double conf = 1.0;
            const std::string res = ctc_beam_dec_bw == 0
                    ? CTCGreedyDecoder(data, sz, symbol_set, pad_symbol, &conf)
                    : CTCBeamSearchDecoder(data, sz, symbol_set, &conf, ctc_beam_dec_bw);
            return {res, conf};
        }
    };

} // namespace ocr
