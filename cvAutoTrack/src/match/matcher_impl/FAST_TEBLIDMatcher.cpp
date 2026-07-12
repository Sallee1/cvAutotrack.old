#include "pch.h"
#include "FAST_TEBLIDMatcher.h"

bool FAST_TEBLIDMatcher::detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	cv::FAST(img, keypoints, 16, true);
	return true;
}

bool FAST_TEBLIDMatcher::compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	teblid->compute(img, keypoints, descriptors);
	return true;
}

bool FAST_TEBLIDMatcher::detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	// 金字塔由基类处理，_impl 只负责单尺度
	detect_impl(img, keypoints);
	if (keypoints.empty()) return false;
	compute_impl(img, keypoints, descriptors);
	return true;
}

cv::Ptr<cv::Feature2D> FAST_TEBLIDMatcher::getFeature2D()
{
	return teblid;
}

bool FAST_TEBLIDMatcher::getIsBinaryDescriptor()
{
	// TEBLID 使用二值描述子
	return true;
}
