// MAC-COMPAT shim: SparseTrack python_module.cpp GMC() body VERBATIM, exposed
// through a C ABI (ctypes) instead of Boost.Python/pbcvt (which only converts
// numpy <-> cv::Mat). No algorithmic change.
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videostab.hpp"
#include <cstring>
#include <iostream>

static cv::Mat GMC(cv::Mat currFrame, cv::Mat prevFrame, int downscale){
    cv::Mat frame, preframe;
    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> est = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(cv::videostab::MM_SIMILARITY);
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(est);
    cv::Size downSize(currFrame.cols / downscale, currFrame.rows / downscale);
    cv::resize(currFrame, frame, downSize, cv::INTER_LINEAR);//
    cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
    if (!prevFrame.empty())
    {
        cv::resize(prevFrame, preframe, downSize, cv::INTER_LINEAR);
        bool ok;
        warp = kbest->estimate(preframe, frame, &ok);
        if (!ok)
        {
            std::cout << "WARNING: Warp not ok" << std::endl;
        }
        warp.convertTo(warp, CV_32F);
        warp.at<float>(0, 2) *= downscale;
        warp.at<float>(1, 2) *= downscale;
    }
    return warp;
}

extern "C" int gmc_c(const unsigned char* curr, const unsigned char* prev,
                     int rows, int cols, int channels, int downscale, float* out9) {
    int type = channels == 3 ? CV_8UC3 : CV_8UC1;
    cv::Mat c(rows, cols, type, (void*)curr);
    cv::Mat p;
    if (prev) p = cv::Mat(rows, cols, type, (void*)prev);
    cv::Mat w = GMC(c.clone(), prev ? p.clone() : cv::Mat(), downscale);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) out9[i * 3 + j] = w.at<float>(i, j);
    return 0;
}
