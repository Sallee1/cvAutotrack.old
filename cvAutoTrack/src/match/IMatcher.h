#pragma once
#include <filesystem>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "algorithm/IBFMatchAlgorithm.h"
#include "algorithm/IIndexedMatchAlgorithm.h"

namespace fs = std::filesystem;

/// 特征匹配编排类
/// 特征提取（detect/compute）和匹配（bf/indexed）均通过依赖注入，
/// 本类负责金字塔多尺度+多亮度增强的编排，以及匹配操作的薄封装委托。
class IMatcher {
public:
	virtual ~IMatcher() = default;

	struct KeyMatPoint
	{
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		bool empty() const { return keypoints.empty(); }
		auto size() const { return keypoints.size(); }
	};

public:
	// --- 算法注入 ---
	/// 设置特征提取器；detector 和 descriptor 必须成对注入
	/// @param detector 特征点检测器
	/// @param descriptor 描述子提取器，默认 nullptr 表示与 detector 共用同一对象
	void setFeature2D(const cv::Ptr<cv::Feature2D>& detector,
					  const cv::Ptr<cv::Feature2D>& descriptor = nullptr)
	{
		m_detector = detector;
		m_descriptor = descriptor ? descriptor : detector;
	}

	const cv::Ptr<cv::Feature2D>& getDetector() const { return m_detector; }
	const cv::Ptr<cv::Feature2D>& getDescriptor() const { return m_descriptor; }

	/// 设置匹配算法；bf 和 indexed 必须成对注入以提供完整匹配功能
	void setMatchAlgorithm(const std::shared_ptr<IBFMatchAlgorithm>& bf,
						   const std::shared_ptr<IIndexedMatchAlgorithm>& indexed)
	{
		m_bf_matcher = bf;
		m_indexed_matcher = indexed;
	}

	const std::shared_ptr<IBFMatchAlgorithm>& getBFMatcher() const { return m_bf_matcher; }
	const std::shared_ptr<IIndexedMatchAlgorithm>& getIndexedMatcher() const { return m_indexed_matcher; }

	// --- 特征提取（单层，不涉及增强） ---
	bool detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

	bool compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	/// 标准单层检测+描述（不使用金字塔/亮度增强）
	bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	bool detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point);

	/// 增强多层检测+描述（金字塔多尺度 × 亮度增益并行处理）
	bool detect_and_compute_ex(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

	bool detect_and_compute_ex(const cv::Mat& img, KeyMatPoint& key_mat_point);

	// --- 索引匹配（委托 m_indexed_matcher） ---
	void cache_train_descriptors(const cv::Mat& train_descriptors);
	bool try_load_index(const fs::path& path, const cv::Mat& train_descriptors);
	bool save_index(const fs::path& path);

	std::vector<std::vector<cv::DMatch>> indexed_knnmatch(const cv::Mat& query_descriptors, int k = 2);
	std::vector<std::vector<cv::DMatch>> indexed_knnmatch(const KeyMatPoint& query, int k = 2);
	std::vector<cv::DMatch> indexed_match(const cv::Mat& query_descriptors);
	std::vector<cv::DMatch> indexed_match(const KeyMatPoint& query);

	// --- 暴力匹配（委托 m_bf_matcher） ---
	std::vector<std::vector<cv::DMatch>> bf_knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k = 2);
	std::vector<std::vector<cv::DMatch>> bf_knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k = 2);
	std::vector<cv::DMatch> bf_match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors);
	std::vector<cv::DMatch> bf_match(const KeyMatPoint& query, const KeyMatPoint& train);

	// --- 金字塔多尺度支持 ---
	void setPyramidScales(const std::vector<double>& scales) { m_pyramid_scales = scales; }
	const std::vector<double>& getPyramidScales() const { return m_pyramid_scales; }

	// --- 多亮度增强支持 ---
	void setBrightnessGains(const std::vector<double>& gains) { m_brightness_gains = gains; }
	const std::vector<double>& getBrightnessGains() const { return m_brightness_gains; }

private:
	cv::Ptr<cv::Feature2D> m_detector;
	cv::Ptr<cv::Feature2D> m_descriptor;
	std::shared_ptr<IBFMatchAlgorithm> m_bf_matcher;
	std::shared_ptr<IIndexedMatchAlgorithm> m_indexed_matcher;

	std::vector<double> m_pyramid_scales;    // 空 = 单尺度
	std::vector<double> m_brightness_gains;  // 空 = 不增强
};