#pragma once
#include <opencv2/opencv.hpp>

/// 暴力匹配算法接口
class IBFMatchAlgorithm {
public:
	virtual ~IBFMatchAlgorithm() = default;

	/// kNN 搜索
	virtual std::vector<std::vector<cv::DMatch>> knnmatch(
		const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k = 2) = 0;

	/// 最近邻搜索
	virtual std::vector<cv::DMatch> match(
		const cv::Mat& query_descriptors, const cv::Mat& train_descriptors) = 0;
};
