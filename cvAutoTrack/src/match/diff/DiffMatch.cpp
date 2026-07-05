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
    result -= cv::Point2d{ 0.5, 0.5 };
    return result;
}
