#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class IMatcher {
public:

	virtual ~IMatcher() = default;
	struct KeyMatPoint
	{
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		bool empty() { return keypoints.size() == 0; }
		auto size() { return keypoints.size(); }
	};

public:
	virtual std::vector<std::vector<cv::DMatch>> knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k = 2, bool bfmatch = false);

	virtual std::vector<std::vector<cv::DMatch>> knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k = 2, bool bfmatch = false);

	virtual std::vector<cv::DMatch> match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool bfmatch = false, bool cross_check = false);

	virtual std::vector<cv::DMatch> match(const KeyMatPoint& query, const KeyMatPoint& train, bool bfmatch = false, bool cross_check = false);

	virtual bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	virtual bool detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point);

	virtual cv::Ptr<cv::Feature2D> getFeature2D() = 0;

	virtual bool getIsBinaryDescriptor() = 0;

protected:
	cv::Ptr<cv::DescriptorMatcher> getMatcher(bool bfmatch = false, bool cross_check = false);
};