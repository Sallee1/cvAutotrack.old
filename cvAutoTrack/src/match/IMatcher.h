#pragma once
#include <vector>
#include <mutex>
#include <memory>
#include <opencv2/opencv.hpp>

class FlannIndex;

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
	// --- FLANN 索引（依赖注入，支持多实例共享）---
	void setFlannIndex(const std::shared_ptr<FlannIndex>& index) { m_flann_index = index; }
	const std::shared_ptr<FlannIndex>& getFlannIndex() const { return m_flann_index; }

	void cache_flann_train_descriptors(const cv::Mat& train_descriptors);
	bool try_load_flann_index(const std::string& path, const cv::Mat& train_descriptors);
	bool save_flann_index(const std::string& path);

	std::vector<std::vector<cv::DMatch>> flann_knnmatch(const cv::Mat& query_descriptors, int k = 2);
	std::vector<std::vector<cv::DMatch>> flann_knnmatch(const KeyMatPoint& query, int k = 2);
	std::vector<cv::DMatch> flann_match(const cv::Mat& query_descriptors);
	std::vector<cv::DMatch> flann_match(const KeyMatPoint& query);

	std::vector<std::vector<cv::DMatch>> bf_knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k = 2);
	std::vector<std::vector<cv::DMatch>> bf_knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k = 2);
	std::vector<cv::DMatch> bf_match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool cross_check = false);
	std::vector<cv::DMatch> bf_match(const KeyMatPoint& query, const KeyMatPoint& train, bool cross_check = false);

	// --- 公开接口（非虚，自带金字塔多尺度） ---
	bool detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

	bool compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	bool detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point);

	// --- 金字塔多尺度支持 ---
	/// 设置图像金字塔缩放比例列表，detect_and_compute 将依次在各尺度提取后合并
	void setPyramidScales(const std::vector<double>& scales) { m_pyramid_scales = scales; }
	const std::vector<double>& getPyramidScales() const { return m_pyramid_scales; }

	// --- 多亮度增强支持 ---
	/// 设置亮度增益列表，detect_and_compute 将在各增益下提取特征后合并（解决亮度敏感）
	void setBrightnessGains(const std::vector<double>& gains) { m_brightness_gains = gains; }
	const std::vector<double>& getBrightnessGains() const { return m_brightness_gains; }

protected:
	/// 子类覆写此方法实现单尺度检测（无需关心金字塔）
	virtual bool detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

	/// 子类覆写此方法实现单尺度描述子计算（无需关心金字塔）
	virtual bool compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	/// 子类可选择性覆写此方法优化单尺度检测+描述（默认调用 detect_impl + compute_impl）
	virtual bool detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	virtual cv::Ptr<cv::Feature2D> getFeature2D() = 0;

	virtual bool getIsBinaryDescriptor() = 0;

private:
	std::vector<double> m_pyramid_scales;    // 空 = 单尺度
	std::vector<double> m_brightness_gains;  // 空 = 不增强
	std::shared_ptr<FlannIndex> m_flann_index;
	cv::Ptr<cv::DescriptorMatcher> create_bf_matcher(bool cross_check = false);
};