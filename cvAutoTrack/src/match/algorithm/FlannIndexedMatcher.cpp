#include "pch.h"
#include "FlannIndexedMatcher.h"

void FlannIndexedMatcher::build(const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return;

	const bool cache_hit = m_index
		&& m_train_descriptors.data == train_descriptors.data
		&& m_train_descriptors.rows == train_descriptors.rows
		&& m_train_descriptors.cols == train_descriptors.cols
		&& m_train_descriptors.type() == train_descriptors.type();

	if (cache_hit) return;

	auto idx = std::make_unique<cv::flann::Index>();
	if (m_is_binary)
		idx->build(train_descriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
	else
		idx->build(train_descriptors, cv::flann::KDTreeIndexParams(8), cvflann::FLANN_DIST_L2);

	m_index = std::move(idx);
	m_train_descriptors = train_descriptors;
}

bool FlannIndexedMatcher::try_load(const fs::path& path, const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return false;

	const bool cache_hit = m_index
		&& m_train_descriptors.data == train_descriptors.data
		&& m_train_descriptors.rows == train_descriptors.rows
		&& m_train_descriptors.cols == train_descriptors.cols
		&& m_train_descriptors.type() == train_descriptors.type();
	if (cache_hit) return true;

	try
	{
		auto idx = std::make_unique<cv::flann::Index>();
		bool loaded = idx->load(train_descriptors, path.u8string());
		if (loaded)
		{
			m_index = std::move(idx);
			m_train_descriptors = train_descriptors;
			return true;
		}
	}
	catch (...)
	{
	}
	return false;
}

bool FlannIndexedMatcher::save(const fs::path& path)
{
	if (!m_index) return false;

	try
	{
		m_index->save(path.u8string());
		return true;
	}
	catch (...)
	{
		return false;
	}
}

std::vector<std::vector<cv::DMatch>> FlannIndexedMatcher::knnmatch(const cv::Mat& query_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty()) return match_group;

	if (!m_index) return match_group;

	cv::Mat indices, dists;
	if (m_is_binary)
		m_index->knnSearch(query_descriptors, indices, dists, k, cv::flann::SearchParams(256, 0.1f, true));
	else
		m_index->knnSearch(query_descriptors, indices, dists, k, cv::flann::SearchParams(256, 0.0f, true));

	match_group.resize(query_descriptors.rows);
	for (int i = 0; i < query_descriptors.rows; ++i)
	{
		match_group[i].reserve(k);
		for (int j = 0; j < k; ++j)
		{
			int train_idx = indices.at<int>(i, j);
			float dist = dists.at<float>(i, j);
			if (train_idx >= 0)
				match_group[i].emplace_back(i, train_idx, dist);
		}
	}
	return match_group;
}

std::vector<cv::DMatch> FlannIndexedMatcher::match(const cv::Mat& query_descriptors)
{
	std::vector<cv::DMatch> matches;
	if (query_descriptors.empty()) return matches;

	if (!m_index) return matches;

	cv::Mat indices, dists;
	if (m_is_binary)
		m_index->knnSearch(query_descriptors, indices, dists, 1, cv::flann::SearchParams(256, 0.1f, true));
	else
		m_index->knnSearch(query_descriptors, indices, dists, 1, cv::flann::SearchParams(256, 0.0f, true));

	matches.reserve(query_descriptors.rows);
	for (int i = 0; i < query_descriptors.rows; ++i)
	{
		int train_idx = indices.at<int>(i, 0);
		if (train_idx >= 0)
			matches.emplace_back(i, train_idx, dists.at<float>(i, 0));
	}
	return matches;
}


