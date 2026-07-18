#pragma once
#include <opencv2/opencv.hpp>

/// 3-DOF 缩放+平移模型求解器
///
/// 模型: dst = s * src + (dx, dy)
/// 相比 estimateAffinePartial2D (4-DOF 相似变换) 去掉旋转自由度，
/// 过定程度提升 33%，适用于无旋转的小地图匹配场景。
///
/// 用法:
///   std::vector<uchar> inliers;
///   cv::Mat H = LinearSolve::estimateScaleTranslation(src, dst, &inliers);
///   if (H.empty()) { /* 失败 */ }
///   cv::Point2d center = /* ... */;
///   cv::transform(std::vector{center}, center, H);
namespace LinearSolve {

/// RANSAC + LS 精化参数
struct STParams
{
	double ransac_thresh = 3.0;       // 内点距离阈值（像素）
	int    ransac_iter   = 500;       // RANSAC 迭代次数
	double min_scale     = 0.31;      // 缩放下限
	double max_scale     = 1.3;       // 缩放上限
	int    min_inliers   = 6;         // 最小内点数
	double min_inlier_ratio = 0.3;    // 最小内点比例
};

/// @brief 估计 3-DOF 缩放+平移变换（RANSAC + 最小二乘精化）
/// @param  src     源点坐标（e.g. 小地图特征点）
/// @param  dst     目标点坐标（e.g. 大地图特征点）
/// @param  inliers [out] 内点掩码（可选），标记参与 LS 精化的点
/// @param  params  求解参数（可选）
/// @return 成功时返回 2×3 CV_64F 矩阵 [s, 0, dx; 0, s, dy]，失败返回空矩阵
cv::Mat estimateScaleTranslation(
	const std::vector<cv::Point2d>& src,
	const std::vector<cv::Point2d>& dst,
	std::vector<uchar>* inliers = nullptr,
	const STParams& params = {});

/// @brief 从 2×3 变换矩阵中提取缩放+平移参数
/// @param H  2×3 矩阵（由 estimateScaleTranslation 返回）
/// @param s  [out] 缩放因子
/// @param dx [out] X 平移量
/// @param dy [out] Y 平移量
/// @return 矩阵是否为有效的缩放+平移模式
bool decomposeST(const cv::Mat& H, double& s, double& dx, double& dy);

} // namespace LinearSolve
