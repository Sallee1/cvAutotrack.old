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
	virtual void cache_flann_train_descriptors(const cv::Mat& train_descriptors);

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
	cv::Ptr<cv::DescriptorMatcher> create_flann_matcher();
	cv::Ptr<cv::DescriptorMatcher> get_cached_flann_matcher();

	std::mutex m_flann_matcher_mutex;
	cv::Ptr<cv::DescriptorMatcher> m_cached_flann_matcher;
	cv::Mat m_cached_flann_train_descriptors;
	bool m_cached_flann_is_binary_descriptor = false;
};