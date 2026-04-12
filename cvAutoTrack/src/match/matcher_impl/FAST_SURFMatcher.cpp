#include "pch.h"
#include "FAST_SURFMatcher.h"

bool FAST_SURFMatcher::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	cv::FAST(img,keypoints,16,true);
	return true;
}

bool FAST_SURFMatcher::compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	detector->compute(img,keypoints,descriptors);
	return true;
}

bool FAST_SURFMatcher::detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	detect(img, keypoints);
	compute(img, keypoints, descriptors);
	return true;
}


cv::Ptr<cv::Feature2D> FAST_SURFMatcher::getFeature2D()
{
	return detector;
}

bool FAST_SURFMatcher::getIsBinaryDescriptor()
{
	return false;
}