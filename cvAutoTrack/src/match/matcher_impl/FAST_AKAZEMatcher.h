#pragma once
#include <match/IMatcher.h>

class FAST_AKAZEMatcher : public IMatcher {
private:
	cv::Ptr<cv::AKAZE> descriptor_extractor;

public:
	FAST_AKAZEMatcher(cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB,
		int descriptor_size = 0, int descriptor_channels = 3,
		float threshold = 0.001f, int nOctaves = 4,
		int nOctaveLayers = 4, cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2,
		int max_points = -1)
	{
		descriptor_extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
			threshold, nOctaves, nOctaveLayers, diffusivity, max_points);
	}
	virtual ~FAST_AKAZEMatcher() = default;

	// 通过 IMatcher 继承（_impl 系列由基类金字塔调用）
	virtual bool detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) override;

	virtual bool compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

	virtual bool detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

	cv::Ptr<cv::Feature2D> getFeature2D() override;
	bool getIsBinaryDescriptor() override;
};
