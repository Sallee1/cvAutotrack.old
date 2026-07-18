#pragma once
#include "IIndexedMatchAlgorithm.h"
#include <memory>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/// OpenCV FLANN 索引匹配器，实现 IIndexedMatchAlgorithm 接口
/// 支持注入到多个 IMatcher 实例共享
class FlannIndexedMatcher : public IIndexedMatchAlgorithm {
public:
	/// @param is_binary 是否为二值描述子（决定 LSH/KDTree 索引类型和距离度量）
	FlannIndexedMatcher(bool is_binary)
		: m_is_binary(is_binary) {}

	void build(const cv::Mat& train_descriptors) override;
	bool try_load(const fs::path& path, const cv::Mat& train_descriptors) override;
	bool save(const fs::path& path) override;

	std::vector<std::vector<cv::DMatch>> knnmatch(const cv::Mat& query_descriptors, int k = 2) override;
	std::vector<cv::DMatch> match(const cv::Mat& query_descriptors) override;

	bool empty() const override { return !m_index; }

private:
	std::unique_ptr<cv::flann::Index> m_index;
	cv::Mat m_train_descriptors;
	bool m_is_binary;
};
