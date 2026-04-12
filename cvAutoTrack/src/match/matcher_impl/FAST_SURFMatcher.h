#pragma once
#include <match/IMatcher.h>
#include <opencv2/xfeatures2d.hpp>

class FAST_SURFMatcher :public IMatcher {
private:
	cv::Ptr<cv::xfeatures2d::SURF> detector;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	bool is_bf_matcher = true;

public:
	FAST_SURFMatcher(double hessianThreshold = 100,
		int nOctaves = 4, int nOctaveLayers = 3,
		bool extended = false, bool upright = false)
	{
		detector = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
	}
	virtual ~FAST_SURFMatcher() = default;

	// 通过 IMatcher 继承
	virtual bool detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

	virtual bool compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	virtual bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	cv::Ptr<cv::Feature2D> getFeature2D() override;
	bool getIsBinaryDescriptor() override;
};