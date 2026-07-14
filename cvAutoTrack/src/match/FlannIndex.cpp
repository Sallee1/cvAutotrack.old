#include "pch.h"
#include "FlannIndex.h"

void FlannIndex::build(const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return;

	std::lock_guard<std::mutex> lock(m_mutex);
	const bool cache_hit = !m_index.empty()
		&& m_train_descriptors.data == train_descriptors.data
		&& m_train_descriptors.rows == train_descriptors.rows
		&& m_train_descriptors.cols == train_descriptors.cols
		&& m_train_descriptors.type() == train_descriptors.type();

	if (cache_hit) return;

	auto idx = create_flann_index();
	if (m_is_binary)
		idx->build(train_descriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
	else
		idx->build(train_descriptors, cv::flann::KDTreeIndexParams(8), cvflann::FLANN_DIST_L2);

	m_index = idx;
	m_train_descriptors = train_descriptors;
}

bool FlannIndex::try_load(const fs::path& path, const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return false;

	std::lock_guard<std::mutex> lock(m_mutex);
	const bool cache_hit = !m_index.empty()
		&& m_train_descriptors.data == train_descriptors.data
		&& m_train_descriptors.rows == train_descriptors.rows
		&& m_train_descriptors.cols == train_descriptors.cols
		&& m_train_descriptors.type() == train_descriptors.type();
	if (cache_hit) return true;

	try
	{
		cv::Ptr<cv::flann::Index> idx = cv::makePtr<cv::flann::Index>();
		bool loaded = idx->load(train_descriptors, path.u8string());
		if (loaded)
		{
			m_index = idx;
			m_train_descriptors = train_descriptors;
			return true;
		}
	}
	catch (...)
	{
	}
	return false;
}

bool FlannIndex::save(const fs::path& path)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	if (m_index.empty()) return false;

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

std::vector<std::vector<cv::DMatch>> FlannIndex::knnmatch(const cv::Mat& query_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty()) return match_group;

	auto index = get_cached_index();
	if (index.empty()) return match_group;

	cv::Mat indices, dists;
	if (m_is_binary)
		index->knnSearch(query_descriptors, indices, dists, k, cv::flann::SearchParams(256, 0.1f, true));
	else
		index->knnSearch(query_descriptors, indices, dists, k, cv::flann::SearchParams(256, 0.0f, true));

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

std::vector<cv::DMatch> FlannIndex::match(const cv::Mat& query_descriptors)
{
	std::vector<cv::DMatch> matches;
	if (query_descriptors.empty()) return matches;

	auto index = get_cached_index();
	if (index.empty()) return matches;

	cv::Mat indices, dists;
	if (m_is_binary)
		index->knnSearch(query_descriptors, indices, dists, 1, cv::flann::SearchParams(256, 0.1f, true));
	else
		index->knnSearch(query_descriptors, indices, dists, 1, cv::flann::SearchParams(256, 0.0f, true));

	matches.reserve(query_descriptors.rows);
	for (int i = 0; i < query_descriptors.rows; ++i)
	{
		int train_idx = indices.at<int>(i, 0);
		if (train_idx >= 0)
			matches.emplace_back(i, train_idx, dists.at<float>(i, 0));
	}
	return matches;
}

cv::Ptr<cv::flann::Index> FlannIndex::create_flann_index()
{
	return cv::makePtr<cv::flann::Index>();
}

cv::Ptr<cv::flann::Index> FlannIndex::get_cached_index()
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_index;
}
