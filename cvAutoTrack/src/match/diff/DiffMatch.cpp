#include "pch.h"
#include "DiffMatch.h"

cv::Point2d DiffMatch::phaseCorrelate(const cv::Mat& ref, const cv::Mat& cur, double& response)
{
    if (ref.size() != cur.size())
    {
        response = 0.0;
        return cv::Point2d(NAN, NAN);
    }
    cv::Mat ref32, cur32;
    ref.convertTo(ref32, CV_32FC1);
    cur.convertTo(cur32, CV_32FC1);

    cv::Mat window;
    cv::createHanningWindow(window, ref32.size(), CV_32FC1);

    auto result = cv::phaseCorrelate(cur32, ref32, window, &response);

    // phaseCorrelate 内部用 getOptimalDFTSize 对图像做 padding，
    // padding 后的尺寸若为奇数，fftShift 将 DC 搬至 (N+1)/2 像素处，
    // 而 center = N/2.0，导致零位移时出现 -0.5 的系统偏置。
    const int optH = cv::getOptimalDFTSize(ref32.rows);
    const int optW = cv::getOptimalDFTSize(ref32.cols);
    cv::Point2d zeroBias(0.0, 0.0);
    if (optW % 2 == 1) zeroBias.x = -0.5;
    if (optH % 2 == 1) zeroBias.y = -0.5;
    result -= zeroBias;

    return result;
}
