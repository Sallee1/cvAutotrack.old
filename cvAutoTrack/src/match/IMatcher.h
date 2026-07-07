#pragma once
#include <vector>
#include <mutex>
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
	/// 构建或更新内存中的 FLANN 索引（从训练描述子）
	virtual void cache_flann_train_descriptors(const cv::Mat& train_descriptors);

	/// 尝试从磁盘加载已缓存的 FLANN 索引；失败时返回 false，调用方应使用 cache_flann_train_descriptors 重建
	virtual bool try_load_flann_index(const std::string& path, const cv::Mat& train_descriptors);

	/// 将当前内存中的 FLANN 索引保存到磁盘
	virtual bool save_flann_index(const std::string& path);

	virtual std::vector<std::vector<cv::DMatch>> flann_knnmatch(const cv::Mat& query_descriptors, int k = 2);

	virtual std::vector<std::vector<cv::DMatch>> flann_knnmatch(const KeyMatPoint& query, int k = 2);

	virtual std::vector<cv::DMatch> flann_match(const cv::Mat& query_descriptors);

	virtual std::vector<cv::DMatch> flann_match(const KeyMatPoint& query);

	virtual std::vector<std::vector<cv::DMatch>> bf_knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k = 2);

	virtual std::vector<std::vector<cv::DMatch>> bf_knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k = 2);

	virtual std::vector<cv::DMatch> bf_match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool cross_check = false);

	virtual std::vector<cv::DMatch> bf_match(const KeyMatPoint& query, const KeyMatPoint& train, bool cross_check = false);

	virtual bool detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

	virtual bool compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	virtual bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	virtual bool detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point);

	virtual cv::Ptr<cv::Feature2D> getFeature2D() = 0;

	virtual bool getIsBinaryDescriptor() = 0;

private:
	cv::Ptr<cv::DescriptorMatcher> create_bf_matcher(bool cross_check = false);
	cv::Ptr<cv::flann::Index> create_flann_index();
	cv::Ptr<cv::flann::Index> get_cached_flann_index();

	mutable std::mutex m_flann_matcher_mutex;
	cv::Ptr<cv::flann::Index> m_cached_flann_index;
	cv::Mat m_cached_flann_train_descriptors;
	bool m_cached_flann_is_binary_descriptor = false;
};