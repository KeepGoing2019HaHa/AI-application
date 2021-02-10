// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>
#include "iostream"

float get_max(ncnn::Mat out) {
    int out_c = out.c;
    int out_w = out.w;
    int out_h = out.h;
    for(int i=0; i<out_c; i++) {
        float max = -100, min = 9999999999999;
        for(int j=0; j<out_h; j++) {
            for(int k=0; k<out_w; k++) {
                if(out.channel(i).row(j)[k] > max) {
                    max = out.channel(i).row(j)[k];
                }
                if(out.channel(i).row(j)[k] < min) {
                    min = out.channel(i).row(j)[k];
                }
            }
        }
        std::cout << max << " " << min << std::endl;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
//    cv::Mat m = cv::imread(imagepath, 1);
    cv::Mat m = cv::imread(imagepath);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }


    ncnn::Net deoldify;
//    deoldify.load_param("/Users/yiwei/Desktop/learn/DeOldify/deoldify.param");
//    deoldify.load_model("/Users/yiwei/Desktop/learn/DeOldify/deoldify.bin");
//    deoldify.load_param("/Users/yiwei/Desktop/learn/DeOldify/deoldify.128.param");
//    deoldify.load_model("/Users/yiwei/Desktop/learn/DeOldify/deoldify.128.bin");
    deoldify.load_param("/Users/yiwei/Desktop/learn/DeOldify/deoldify.256.param");
    deoldify.load_model("/Users/yiwei/Desktop/learn/DeOldify/deoldify.256.bin");

//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_RGB, m.cols, m.rows, 512, 512);
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_RGB, m.cols, m.rows, 128, 128);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_RGB, m.cols, m.rows, 256, 256);
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR, m.cols, m.rows, 128, 128);
//    const float pre_mean_vals[3] = {0.f, 0.f, 0.f};
//    const float pre_norm_vals[3] = {1.0/255.0f,1.0/255.0f,1.0/255.0f};
//    in.substract_mean_normalize(pre_mean_vals, pre_norm_vals);
//    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
//    const float norm_vals[3] = {0.229f, 0.224f, 0.225f};
//    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = deoldify.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
//    ex.extract("553", out);
//    ex.extract("608", out);
//    ex.extract("611", out);
//    ex.extract("612", out);
//    ex.extract("613", out);
//    std::cout << out.channel(0).row(0)[0] << out.channel(0).row(0)[10] << out.channel(0).row(0)[20];
    ex.extract("out", out);
    get_max(out);


//    cv::Mat cv_out = cv::Mat::zeros(512, 512, CV_8UC3);
//    cv::Mat cv_out = cv::Mat::zeros(128, 128, CV_8UC3);
    cv::Mat cv_out = cv::Mat::zeros(256, 256, CV_8UC3);

//    out.to_pixels(cv_out.data, ncnn::Mat::PIXEL_RGB);
//    out.to_pixels(cv_out.data, ncnn::Mat::PIXEL_BGR);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < out.h; i++) {
            for (int j = 0; j < out.w; j++) {
                float t = ((float*)out.data)[j + i * out.w + c * out.h * out.w];
                cv_out.data[(2 - c) + j * 3 + i * out.w * 3] = t;
            }
        }
    }


    cv::imwrite("ncnn.jpg", cv_out);


    fprintf(stderr, "done");

    return 0;
}

