#include "pch.h"
#include "OpenCVBFMatcher.h"

std::vector<std::vector<cv::DMatch>> OpenCVBFMatcher::knnmatch(
	const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty() || train_descriptors.empty())
		return match_group;

	m_knn_matcher->knnMatch(query_descriptors, train_descriptors, match_group, k);
	return match_group;
}

std::vector<cv::DMatch> OpenCVBFMatcher::match(
	const cv::Mat& query_descriptors, const cv::Mat& train_descriptors)
{
	std::vector<cv::DMatch> matches;
	if (query_descriptors.empty() || train_descriptors.empty())
		return matches;

	m_match_matcher->match(query_descriptors, train_descriptors, matches);
	return matches;
}
