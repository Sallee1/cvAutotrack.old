#pragma once
#include "IBFMatchAlgorithm.h"
#include <opencv2/opencv.hpp>

/// OpenCV 暴力匹配器封装
class OpenCVBFMatcher : public IBFMatchAlgorithm {
public:
	/// @param norm_type 距离度量类型（cv::NORM_HAMMING / cv::NORM_L2）
	/// @param cross_check 是否启用交叉验证
	OpenCVBFMatcher(int norm_type, bool cross_check = false)
		: m_norm_type(norm_type), m_cross_check(cross_check)
	{
		m_knn_matcher = cv::BFMatcher::create(norm_type, false);
		m_match_matcher = cv::BFMatcher::create(norm_type, cross_check);
	}

	std::vector<std::vector<cv::DMatch>> knnmatch(
		const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k = 2) override;

	std::vector<cv::DMatch> match(
		const cv::Mat& query_descriptors, const cv::Mat& train_descriptors) override;

private:
	int m_norm_type;
	bool m_cross_check;
	cv::Ptr<cv::DescriptorMatcher> m_knn_matcher;
	cv::Ptr<cv::DescriptorMatcher> m_match_matcher;
};
