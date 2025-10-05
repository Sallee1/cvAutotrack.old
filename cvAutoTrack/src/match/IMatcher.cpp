#include "pch.h"
#include "IMatcher.h"

std::vector<std::vector<cv::DMatch>> IMatcher::knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k, bool bfmatch) {
	std::vector<std::vector<cv::DMatch>> match_group;
	auto matcher = getMatcher(bfmatch, false);
	matcher->knnMatch(query_descriptors, train_descriptors, match_group, k);
	return match_group;
}

std::vector<std::vector<cv::DMatch>> IMatcher::knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k, bool bfmatch) {
	return knnmatch(query.descriptors, train.descriptors, k, bfmatch);
}

std::vector<cv::DMatch> IMatcher::match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool bfmatch, bool cross_check) {
	std::vector<cv::DMatch> matches;
	auto matcher = getMatcher(bfmatch, cross_check);
	matcher->match(query_descriptors, train_descriptors, matches);
	return matches;
}

std::vector<cv::DMatch> IMatcher::match(const KeyMatPoint& query, const KeyMatPoint& train, bool bfmatch, bool cross_check) {
	return match(query.descriptors, train.descriptors, bfmatch, cross_check);
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

cv::Ptr<cv::DescriptorMatcher> IMatcher::getMatcher(bool bfmatch, bool cross_check)
{
	cv::Ptr<cv::DescriptorMatcher> matcher;
	if (bfmatch)
	{
		if (getIsBinaryDescriptor())
		{
			matcher = cv::BFMatcher::create(cv::NORM_HAMMING, cross_check);
		}
		else {
			matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
		}
	}
	else
	{
		if (getIsBinaryDescriptor())
		{
			cv::Ptr<cv::flann::LshIndexParams> index_params{ cv::makePtr<cv::flann::LshIndexParams>(20, 32, 2) };
			cv::Ptr<cv::flann::SearchParams> search_params{ cv::makePtr<cv::flann::SearchParams>(32, 0.1, true) };
			matcher = cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
		}
		else
		{
			cv::Ptr<cv::flann::KDTreeIndexParams> index_params{ cv::makePtr<cv::flann::KDTreeIndexParams>(6) };
			cv::Ptr<cv::flann::SearchParams> search_params{ cv::makePtr<cv::flann::SearchParams>(32,0, true) };
			matcher = cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
		}
	}
	return matcher;
}