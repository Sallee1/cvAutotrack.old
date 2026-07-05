#pragma once
#include <opencv2/opencv.hpp>

/// 纯静态工具：相位相关计算帧间平移位移
namespace DiffMatch
{
	/// 互相关峰值低于此值认为不可信
	constexpr double CONFIDENCE_THRESHOLD = 0.30;

	/**
	 * @brief 计算两帧之间的平移位移
	 * @param ref  参考帧（旧帧）
	 * @param cur  当前帧（新帧）
	 * @param response [out] 互相关峰值 [0,1]
	 * @return cur 相对于 ref 的位移 (cur - ref)，即参考帧→当前帧的偏移
	 */
	cv::Point2d phaseCorrelate(const cv::Mat& ref, const cv::Mat& cur, double& response);
}
