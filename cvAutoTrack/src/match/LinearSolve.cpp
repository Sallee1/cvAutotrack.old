#include "pch.h"
#include "LinearSolve.h"

namespace LinearSolve {
namespace {

// ============================================================================
// 内部最小求解器: 2 对匹配点 → (s, dx, dy)
// ============================================================================

/// @return false 当两点重合或 s <= 0
bool solve_scale_translation_2pair(
	const cv::Point2d& src1, const cv::Point2d& src2,
	const cv::Point2d& dst1, const cv::Point2d& dst2,
	double& s, double& dx, double& dy)
{
	double dsx = src2.x - src1.x;
	double dsy = src2.y - src1.y;
	double ddx = dst2.x - dst1.x;
	double ddy = dst2.y - dst1.y;

	double norm_sq = dsx * dsx + dsy * dsy;
	if (norm_sq < 1e-10)
		return false;

	// 投影: s = (Δd · Δs) / |Δs|²
	s = (ddx * dsx + ddy * dsy) / norm_sq;
	if (s <= 0.0)
		return false;

	dx = dst1.x - s * src1.x;
	dy = dst1.y - s * src1.y;
	return true;
}

// ============================================================================
// 内部最小二乘精化: 所有内点 → (s, dx, dy)
// ============================================================================

void solve_scale_translation_ls(
	const std::vector<cv::Point2d>& src,
	const std::vector<cv::Point2d>& dst,
	double& s, double& dx, double& dy)
{
	const int N = static_cast<int>(src.size());
	// 构建超定方程组 A * [s, dx, dy]^T = b
	// 每对点贡献 2 行：
	//   [src.x, 1, 0] * [s, dx, dy]^T = dst.x
	//   [src.y, 0, 1] * [s, dx, dy]^T = dst.y
	cv::Mat A(N * 2, 3, CV_64F);
	cv::Mat b(N * 2, 1, CV_64F);
	for (int i = 0; i < N; i++) {
		A.at<double>(i * 2,     0) = src[i].x;
		A.at<double>(i * 2,     1) = 1.0;
		A.at<double>(i * 2,     2) = 0.0;
		b.at<double>(i * 2,     0) = dst[i].x;

		A.at<double>(i * 2 + 1, 0) = src[i].y;
		A.at<double>(i * 2 + 1, 1) = 0.0;
		A.at<double>(i * 2 + 1, 2) = 1.0;
		b.at<double>(i * 2 + 1, 0) = dst[i].y;
	}
	cv::Mat x;
	cv::solve(A, b, x, cv::DECOMP_SVD);
	s  = x.at<double>(0);
	dx = x.at<double>(1);
	dy = x.at<double>(2);
}

/// 打包为 2×3 矩阵: [s, 0, dx; 0, s, dy]
cv::Mat pack_ST(double s, double dx, double dy)
{
	cv::Mat H(2, 3, CV_64F);
	H.at<double>(0, 0) = s;  H.at<double>(0, 1) = 0.0; H.at<double>(0, 2) = dx;
	H.at<double>(1, 0) = 0.0; H.at<double>(1, 1) = s;  H.at<double>(1, 2) = dy;
	return H;
}

} // anonymous namespace

// ============================================================================
// 公开 API
// ============================================================================

cv::Mat estimateScaleTranslation(
	const std::vector<cv::Point2d>& src,
	const std::vector<cv::Point2d>& dst,
	std::vector<uchar>* inliers,
	const STParams& params)
{
	const int N = static_cast<int>(src.size());
	if (N < 2)
		return cv::Mat(); // 空矩阵表示失败

	const double thresh_sq = params.ransac_thresh * params.ransac_thresh;

	// --- RANSAC ---
	cv::RNG rng(cv::getTickCount());
	double best_s = 1.0, best_dx = 0.0, best_dy = 0.0;
	int best_inliers = 0;

	for (int iter = 0; iter < params.ransac_iter; iter++) {
		int i1 = rng.uniform(0, N);
		int i2 = rng.uniform(0, N);
		if (i1 == i2) continue;

		double s, dx, dy;
		if (!solve_scale_translation_2pair(src[i1], src[i2],
		                                   dst[i1], dst[i2],
		                                   s, dx, dy))
			continue;
		if (s < params.min_scale || s > params.max_scale)
			continue;

		int count = 0;
		for (int k = 0; k < N; k++) {
			double ex = dst[k].x - (s * src[k].x + dx);
			double ey = dst[k].y - (s * src[k].y + dy);
			if (ex * ex + ey * ey < thresh_sq)
				count++;
		}
		if (count > best_inliers) {
			best_inliers = count;
			best_s = s; best_dx = dx; best_dy = dy;
		}
	}

	// --- 内点检查 ---
	if (best_inliers < params.min_inliers ||
	    static_cast<double>(best_inliers) / N < params.min_inlier_ratio)
	{
		return cv::Mat();
	}

	// --- 收集内点 → 最小二乘精化 ---
	std::vector<cv::Point2d> inlier_src, inlier_dst;
	inlier_src.reserve(best_inliers);
	inlier_dst.reserve(best_inliers);

	if (inliers) {
		inliers->assign(N, 0);
	}

	for (int k = 0; k < N; k++) {
		double ex = dst[k].x - (best_s * src[k].x + best_dx);
		double ey = dst[k].y - (best_s * src[k].y + best_dy);
		if (ex * ex + ey * ey < thresh_sq) {
			inlier_src.push_back(src[k]);
			inlier_dst.push_back(dst[k]);
			if (inliers) (*inliers)[k] = 1;
		}
	}

	double final_s = best_s, final_dx = best_dx, final_dy = best_dy;
	solve_scale_translation_ls(inlier_src, inlier_dst, final_s, final_dx, final_dy);

	// LS 精化后再次校验
	if (final_s < params.min_scale || final_s > params.max_scale)
		return cv::Mat();

	return pack_ST(final_s, final_dx, final_dy);
}

bool decomposeST(const cv::Mat& H, double& s, double& dx, double& dy)
{
	if (H.empty() || H.rows != 2 || H.cols != 3 || H.type() != CV_64F)
		return false;

	s  = H.at<double>(0, 0);
	dx = H.at<double>(0, 2);
	dy = H.at<double>(1, 2);
	return true;
}

} // namespace LinearSolve
