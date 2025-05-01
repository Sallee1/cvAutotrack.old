#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief 对匹配的图像进行预处理
 * @param in 原图
 * @param out 处理图
 */
inline void matchPreProcess(cv::InputArray in, cv::OutputArray out)
{
    //目前暂时只进行自适应对比度增强处理
    cv::Mat yuv_in;
    cv::cvtColor(in, yuv_in, cv::COLOR_BGR2YUV);

    std::vector<cv::Mat> yuv_channels;
    cv::split(yuv_in, yuv_channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->apply(yuv_channels[0], yuv_channels[0]);
    cv::merge(yuv_channels, yuv_in);

    if (in.cols() != out.cols() && in.rows() != out.rows() && in.type() != out.type())
    {
        out.create(in.size(), in.type());
    }

    cv::cvtColor(yuv_in, out, cv::COLOR_YUV2BGR);
}