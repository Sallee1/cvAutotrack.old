#include "pch.h"
#include "ORBMatcher.h"

bool ORBMatcher::detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	cv::FAST(img, keypoints, 16, true);
	return true;
}

bool ORBMatcher::compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	// 置零方向，得到 upright 版本的 rBRIEF 描述子
	for (auto& kp : keypoints)
	{
		kp.angle = 0;
	}
	orb->compute(img, keypoints, descriptors);
	return true;
}

bool ORBMatcher::detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	// 金字塔由基类 IMatcher::detect_and_compute 处理，_impl 只负责单尺度
	detect_impl(img, keypoints);
	if (keypoints.empty()) return false;
	compute_impl(img, keypoints, descriptors);
	return true;
}

cv::Ptr<cv::Feature2D> ORBMatcher::getFeature2D()
{
	return static_cast<cv::Ptr<cv::Feature2D>>(orb);
}

bool ORBMatcher::getIsBinaryDescriptor()
{
	// ORB 的 rBRIEF 是二值描述子
	return true;
}
