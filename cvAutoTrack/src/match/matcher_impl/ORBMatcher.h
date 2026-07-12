#pragma once
#include <match/IMatcher.h>

class ORBMatcher : public IMatcher {
private:
	cv::Ptr<cv::ORB> orb;

public:
	ORBMatcher(int nfeatures = 100000, float scaleFactor = 1.2f, int nlevels = 1,
		int edgeThreshold = 15, int firstLevel = 0, int WTA_K = 2,
		cv::ORB::ScoreType scoreType = cv::ORB::FAST_SCORE,
		int patchSize = 31, int fastThreshold = 20)
	{
		orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
			firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
	}
	virtual ~ORBMatcher() = default;

	// 通过 IMatcher 继承（_impl 系列由基类金字塔调用）
	virtual bool detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) override;

	virtual bool compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

	virtual bool detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

	cv::Ptr<cv::Feature2D> getFeature2D() override;
	bool getIsBinaryDescriptor() override;
};
