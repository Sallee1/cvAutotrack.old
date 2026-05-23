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
	const bool cache_hit = !m_cached_flann_matcher.empty()
		&& m_cached_flann_train_descriptors.data == train_descriptors.data
		&& m_cached_flann_train_descriptors.rows == train_descriptors.rows
		&& m_cached_flann_train_descriptors.cols == train_descriptors.cols
		&& m_cached_flann_train_descriptors.type() == train_descriptors.type()
		&& m_cached_flann_is_binary_descriptor == is_binary_descriptor;

	if (cache_hit)
	{
		return;
	}

	auto matcher = create_flann_matcher();
	matcher->add(std::vector<cv::Mat>{ train_descriptors });
	matcher->train();

	m_cached_flann_matcher = matcher;
	m_cached_flann_train_descriptors = train_descriptors;
	m_cached_flann_is_binary_descriptor = is_binary_descriptor;
}

std::vector<std::vector<cv::DMatch>> IMatcher::flann_knnmatch(const cv::Mat& query_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty())
	{
		return match_group;
	}

	auto matcher = get_cached_flann_matcher();
	if (matcher.empty())
	{
		return match_group;
	}

	matcher->knnMatch(query_descriptors, match_group, k);
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

	auto matcher = get_cached_flann_matcher();
	if (matcher.empty())
	{
		return matches;
	}

	matcher->match(query_descriptors, matches);
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

cv::Ptr<cv::DescriptorMatcher> IMatcher::create_flann_matcher()
{
	if (getIsBinaryDescriptor())
	{
		cv::Ptr<cv::flann::LshIndexParams> index_params{ cv::makePtr<cv::flann::LshIndexParams>(20, 32, 2) };
		cv::Ptr<cv::flann::SearchParams> search_params{ cv::makePtr<cv::flann::SearchParams>(256, 0.1, true) };
		return cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
	}

	cv::Ptr<cv::flann::KDTreeIndexParams> index_params{ cv::makePtr<cv::flann::KDTreeIndexParams>(8) };
	cv::Ptr<cv::flann::SearchParams> search_params{ cv::makePtr<cv::flann::SearchParams>(256, 0, true) };
	return cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
}

cv::Ptr<cv::DescriptorMatcher> IMatcher::get_cached_flann_matcher()
{
	std::lock_guard<std::mutex> lock(m_flann_matcher_mutex);
	return m_cached_flann_matcher;
}
