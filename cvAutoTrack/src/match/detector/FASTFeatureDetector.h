#pragma once
#include <opencv2/opencv.hpp>

/// FAST 角点检测器 — 包装为 cv::Feature2D，支持注入到 IMatcher
class FASTFeatureDetector : public cv::Feature2D {
public:
	/// @param threshold FAST 阈值
	/// @param nonmax_suppression 是否启用非极大值抑制
	FASTFeatureDetector(int threshold = 16, bool nonmax_suppression = true)
		: m_threshold(threshold), m_nonmax_suppression(nonmax_suppression) {}

	/// 设置关键点模板 — FAST 只填充坐标，其余字段（class_id, octave, size, angle 等）使用此模板值
	void setKeyPointTemplate(const cv::KeyPoint& tpl) { m_kp_template = tpl; }
	const cv::KeyPoint& getKeyPointTemplate() const { return m_kp_template; }

protected:
	void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
				cv::InputArray mask = cv::noArray()) override;

	void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
				 cv::OutputArray descriptors) override;

private:
	int m_threshold;
	bool m_nonmax_suppression;
	cv::KeyPoint m_kp_template;  // 默认值由 cv::KeyPoint 默认构造提供
};
