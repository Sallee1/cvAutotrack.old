#include "pch.h"
#include "FASTFeatureDetector.h"

void FASTFeatureDetector::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
								  cv::InputArray mask)
{
	cv::FAST(image, keypoints, m_threshold, m_nonmax_suppression);

	// 应用关键点模板：FAST 只填充坐标，其余属性用模板值覆盖
	if (m_kp_template.size > 0.f || m_kp_template.octave != 0 ||
		m_kp_template.angle != -1.f || m_kp_template.class_id != -1)
	{
		for (auto& kp : keypoints)
		{
			kp.angle = m_kp_template.angle;
			kp.size = m_kp_template.size;
			kp.octave = m_kp_template.octave;
			kp.class_id = m_kp_template.class_id;
		}
	}
}

void FASTFeatureDetector::compute(cv::InputArray /*image*/,
								   std::vector<cv::KeyPoint>& /*keypoints*/,
								   cv::OutputArray /*descriptors*/)
{
	// FAST 不产生描述子，此方法为 no-op
	return;
}
