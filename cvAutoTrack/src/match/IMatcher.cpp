#include "pch.h"
#include "IMatcher.h"

void IMatcher::cache_flann_train_descriptors(const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty())
	{
		return;
	}

	const bool is_binary_descriptor = getIsBinaryDescriptor();
	std::lock_guard<std::mutex> lock(m_flann_matcher_mutex);
	const bool cache_hit = !m_cached_flann_index.empty()
		&& m_cached_flann_train_descriptors.data == train_descriptors.data
		&& m_cached_flann_train_descriptors.rows == train_descriptors.rows
		&& m_cached_flann_train_descriptors.cols == train_descriptors.cols
		&& m_cached_flann_train_descriptors.type() == train_descriptors.type()
		&& m_cached_flann_is_binary_descriptor == is_binary_descriptor;

	if (cache_hit)
	{
		return;
	}

	auto index = create_flann_index();
	if (is_binary_descriptor)
	{
		index->build(train_descriptors, cv::flann::LshIndexParams(20, 32, 2), cvflann::FLANN_DIST_HAMMING);
	}
	else
	{
		index->build(train_descriptors, cv::flann::KDTreeIndexParams(8), cvflann::FLANN_DIST_L2);
	}

	m_cached_flann_index = index;
	m_cached_flann_train_descriptors = train_descriptors;
	m_cached_flann_is_binary_descriptor = is_binary_descriptor;
}

bool IMatcher::try_load_flann_index(const std::string& path, const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty())
		return false;

	const bool is_binary_descriptor = getIsBinaryDescriptor();
	std::lock_guard<std::mutex> lock(m_flann_matcher_mutex);

	// 如果内存缓存已命中，无需加载
	const bool cache_hit = !m_cached_flann_index.empty()
		&& m_cached_flann_train_descriptors.data == train_descriptors.data
		&& m_cached_flann_train_descriptors.rows == train_descriptors.rows
		&& m_cached_flann_train_descriptors.cols == train_descriptors.cols
		&& m_cached_flann_train_descriptors.type() == train_descriptors.type()
		&& m_cached_flann_is_binary_descriptor == is_binary_descriptor;
	if (cache_hit)
		return true;

	try
	{
		cv::Ptr<cv::flann::Index> index = cv::makePtr<cv::flann::Index>();
		bool loaded = index->load(train_descriptors, path);
		if (loaded)
		{
			m_cached_flann_index = index;
			m_cached_flann_train_descriptors = train_descriptors;
			m_cached_flann_is_binary_descriptor = is_binary_descriptor;
			return true;
		}
	}
	catch (...)
	{
		// 加载失败，静默返回 false
	}

	return false;
}

bool IMatcher::save_flann_index(const std::string& path)
{
	std::lock_guard<std::mutex> lock(m_flann_matcher_mutex);
	if (m_cached_flann_index.empty())
		return false;

	try
	{
		m_cached_flann_index->save(path);
		return true;
	}
	catch (...)
	{
		return false;
	}
}

std::vector<std::vector<cv::DMatch>> IMatcher::flann_knnmatch(const cv::Mat& query_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty())
	{
		return match_group;
	}

	auto index = get_cached_flann_index();
	if (index.empty())
	{
		return match_group;
	}

	cv::Mat indices, dists;
	if (getIsBinaryDescriptor())
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
			// FLANN returns -1 for no match found
			if (train_idx >= 0)
				match_group[i].emplace_back(i, train_idx, dist);
		}
	}

	return match_group;
}

std::vector<std::vector<cv::DMatch>> IMatcher::flann_knnmatch(const KeyMatPoint& query, int k)
{
	return flann_knnmatch(query.descriptors, k);
}

std::vector<cv::DMatch> IMatcher::flann_match(const cv::Mat& query_descriptors)
{
	std::vector<cv::DMatch> matches;
	if (query_descriptors.empty())
	{
		return matches;
	}

	auto index = get_cached_flann_index();
	if (index.empty())
	{
		return matches;
	}

	cv::Mat indices, dists;
	if (getIsBinaryDescriptor())
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

std::vector<cv::DMatch> IMatcher::flann_match(const KeyMatPoint& query)
{
	return flann_match(query.descriptors);
}

std::vector<std::vector<cv::DMatch>> IMatcher::bf_knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty() || train_descriptors.empty())
	{
		return match_group;
	}

	auto matcher = create_bf_matcher(false);
	matcher->knnMatch(query_descriptors, train_descriptors, match_group, k);
	return match_group;
}

std::vector<std::vector<cv::DMatch>> IMatcher::bf_knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k)
{
	return bf_knnmatch(query.descriptors, train.descriptors, k);
}

std::vector<cv::DMatch> IMatcher::bf_match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool cross_check)
{
	std::vector<cv::DMatch> matches;
	if (query_descriptors.empty() || train_descriptors.empty())
	{
		return matches;
	}

	auto matcher = create_bf_matcher(cross_check);
	matcher->match(query_descriptors, train_descriptors, matches);
	return matches;
}

std::vector<cv::DMatch> IMatcher::bf_match(const KeyMatPoint& query, const KeyMatPoint& train, bool cross_check)
{
	return bf_match(query.descriptors, train.descriptors, cross_check);
}

bool IMatcher::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	if (img.empty()) return  false;
	auto detector = getFeature2D();
	detector->detect(img, keypoints);
	return true;
}

bool IMatcher::compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty() || keypoints.size() == 0) return  false;
	auto detector = getFeature2D();
	detector->compute(img, keypoints, descriptors);
	return true;
}

bool IMatcher::detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty()) return  false;
	auto detector = getFeature2D();
	detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
	if (keypoints.size() == 0) return false;
	return true;
}

bool IMatcher::detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point) {
	return detect_and_compute(img, key_mat_point.keypoints, key_mat_point.descriptors);
}

cv::Ptr<cv::DescriptorMatcher> IMatcher::create_bf_matcher(bool cross_check)
{
	cv::Ptr<cv::DescriptorMatcher> matcher;
	if (getIsBinaryDescriptor())
	{
		matcher = cv::BFMatcher::create(cv::NORM_HAMMING, cross_check);
	}
	else
	{
		matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
	}
	return matcher;
}

cv::Ptr<cv::flann::Index> IMatcher::create_flann_index()
{
	return cv::makePtr<cv::flann::Index>();
}

cv::Ptr<cv::flann::Index> IMatcher::get_cached_flann_index()
{
	std::lock_guard<std::mutex> lock(m_flann_matcher_mutex);
	return m_cached_flann_index;
}
