#pragma once
#include <match/IMatcher.h>
#include <opencv2/xfeatures2d.hpp>

class FAST_TEBLIDMatcher : public IMatcher {
private:
	cv::Ptr<cv::Feature2D> teblid;

public:
	/// @param scale_factor 采样窗口缩放: 1.0f=ORB, 5.0f=FAST/BRISK, 6.25f=KAZE/SURF/AKAZE
	/// @param n_bits 描述子位数: TEBLID::SIZE_256_BITS(102, 32字节) 或 TEBLID::SIZE_512_BITS(103, 64字节)
	FAST_TEBLIDMatcher(float scale_factor = 5.0f, int n_bits = cv::xfeatures2d::TEBLID::SIZE_256_BITS)
	{
		teblid = cv::xfeatures2d::TEBLID::create(scale_factor, n_bits);
	}
	virtual ~FAST_TEBLIDMatcher() = default;

	// 通过 IMatcher 继承（_impl 系列由基类金字塔调用）
	virtual bool detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) override;

	virtual bool compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

	virtual bool detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override;

	cv::Ptr<cv::Feature2D> getFeature2D() override;
	bool getIsBinaryDescriptor() override;
};
