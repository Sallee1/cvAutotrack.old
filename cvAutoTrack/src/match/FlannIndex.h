#pragma once
#include <mutex>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/// 独立的 FLANN 索引，支持注入到多个 IMatcher 实例共享
class FlannIndex {
public:
	FlannIndex(bool is_binary_descriptor)
		: m_is_binary(is_binary_descriptor) {}

	/// 构建或更新内存中的 FLANN 索引（从训练描述子）
	void build(const cv::Mat& train_descriptors);

	/// 尝试从磁盘加载；失败返回 false
	bool try_load(const fs::path& path, const cv::Mat& train_descriptors);

	/// 保存到磁盘
	bool save(const fs::path& path);

	/// kNN 搜索
	std::vector<std::vector<cv::DMatch>> knnmatch(const cv::Mat& query_descriptors, int k = 2);

	/// 最近邻搜索
	std::vector<cv::DMatch> match(const cv::Mat& query_descriptors);

	bool empty() const { return m_index.empty(); }

private:
	cv::Ptr<cv::flann::Index> create_flann_index();
	cv::Ptr<cv::flann::Index> get_cached_index();

	mutable std::mutex m_mutex;
	cv::Ptr<cv::flann::Index> m_index;
	cv::Mat m_train_descriptors;
	bool m_is_binary;
};
