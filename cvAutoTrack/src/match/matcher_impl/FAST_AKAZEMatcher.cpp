#include "pch.h"
#include "FAST_AKAZEMatcher.h"

bool FAST_AKAZEMatcher::detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	cv::FAST(img, keypoints, 16, true);
	return true;
}

bool FAST_AKAZEMatcher::compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	//按照AKAZE需要的输入调整参数
	std::for_each(std::execution::par_unseq,keypoints.begin(),keypoints.end(),[](cv::KeyPoint& kp){
		kp.class_id = 0;
		kp.octave = 0;
		kp.size = 4.8f;
	});
	descriptor_extractor->compute(img, keypoints, descriptors);
	return true;
}

bool FAST_AKAZEMatcher::detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	detect_impl(img, keypoints);
	compute_impl(img, keypoints, descriptors);
	return true;
}

cv::Ptr<cv::Feature2D> FAST_AKAZEMatcher::getFeature2D()
{
	return descriptor_extractor;
}

bool FAST_AKAZEMatcher::getIsBinaryDescriptor()
{
	// AKAZE MLDB 是二值描述子
	return true;
}
