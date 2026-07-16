#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/// 索引匹配算法接口
class IIndexedMatchAlgorithm {
public:
	virtual ~IIndexedMatchAlgorithm() = default;

	/// 构建索引
	virtual void build(const cv::Mat& train_descriptors) = 0;

	/// 从磁盘加载索引
	virtual bool try_load(const fs::path& path, const cv::Mat& train_descriptors) = 0;

	/// 保存索引到磁盘
	virtual bool save(const fs::path& path) = 0;

	/// kNN 搜索
	virtual std::vector<std::vector<cv::DMatch>> knnmatch(const cv::Mat& query_descriptors, int k = 2) = 0;

	/// 最近邻搜索
	virtual std::vector<cv::DMatch> match(const cv::Mat& query_descriptors) = 0;

	/// 索引是否为空
	virtual bool empty() const = 0;
};
