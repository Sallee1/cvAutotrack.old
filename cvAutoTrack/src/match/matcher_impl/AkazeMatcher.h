#pragma once
#include <match/IMatcher.h>

class AKAZEMatcher :public IMatcher {
private:
	cv::Ptr<cv::AKAZE> detector;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	bool is_binary_descriptor = false;
	bool is_bf_matcher = true;

public:
	AKAZEMatcher(cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB,
		int descriptor_size = 0, int descriptor_channels = 3,
		float threshold = 0.001f, int nOctaves = 4,
		int nOctaveLayers = 4, cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2,
		int max_points = -1)
	{
		detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity, max_points);
		if (descriptor_type == cv::AKAZE::DESCRIPTOR_MLDB || descriptor_type == cv::AKAZE::DESCRIPTOR_MLDB_UPRIGHT)
		{
			is_binary_descriptor = true;
		}
		else {
			is_binary_descriptor = false;
		}
	}
	virtual ~AKAZEMatcher() = default;

	// 通过 IMatcher 继承
	cv::Ptr<cv::Feature2D> getFeature2D() override;
	bool getIsBinaryDescriptor() override;
};