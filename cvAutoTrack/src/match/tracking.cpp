#include "pch.h"
#include "tracking.h"
#include "match/type/MatchType.h"
#include "resources/Resources.h"
#include "resources/KeypointsCache.h"
#include "utils/Utils.h"

void Tracking::setMap(cv::Mat gi_map)
{
	_mapMat = gi_map;
}

void Tracking::setMiniMap(cv::Mat miniMapMat, float diameter)
{
	_miniMapMat = miniMapMat;
	if (diameter > 0)
	{
		_miniMapDiameter = diameter;
	}
	else
	{
		_miniMapDiameter = std::min(miniMapMat.cols, miniMapMat.rows);
	}
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher)
{
	if (isInit)return true;
	m_matcher->detect_and_compute(_mapMat, map_kp.keypoints, map_kp.descriptors);
	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;
	isInit = true;
	return true;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher, std::vector<cv::KeyPoint>&& gi_map_keypoints, cv::Mat&& gi_map_descriptors)
{
	if (isInit)return true;
	map_kp.keypoints = std::move(gi_map_keypoints);
	map_kp.descriptors = std::move(gi_map_descriptors);
	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;

	auto cell_size = Resources::getInstance().lsh_cell_size;
	m_lsh_index = std::make_unique<KeypointGridLSH>();
	m_lsh_index->build(map_kp.keypoints, map_kp.descriptors, cv::Rect2i(0, 0, _mapMat.cols, _mapMat.rows), cv::Size2i(cell_size, cell_size));

	isInit = true;
	return true;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher, MapKeypointCache&& map_keypoints_cache)
{
	if (isInit) return false;
	m_lsh_index = std::make_unique<KeypointGridLSH>();
	m_lsh_index->fromCache(map_keypoints_cache);
	map_kp.keypoints = std::move(map_keypoints_cache.keypoints);
	map_kp.descriptors = std::move(map_keypoints_cache.descriptors);

	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;
	isInit = true;
	return true;
}

void Tracking::UnInit()
{
	if (!isInit)return;
	_mapMat.release();
	_mapMat = cv::Mat();
	map_kp.keypoints.clear();
	map_kp.descriptors.release();
	isInit = false;
}

void Tracking::match()
{
	bool calc_is_faile = false;
	is_success_match = false;

	// 非连续匹配，匹配整个大地图
	if (isMatchAllMap)
	{
		pos = match_no_continuity(calc_is_faile);
		if (std::isnan(pos.x) || std::isnan(pos.y))
		{
			calc_is_faile = true;
		}

		// 没有有效结果，结束
		if (calc_is_faile)
		{
			pos = last_pos;
			is_success_match = false;
			return;
		}
		continuity_retry = max_continuity_retry - 1;		//全局检测后只局部检测一次
	}

	// 尝试连续匹配，匹配角色附近小范围区域
	bool calc_continuity_is_faile = false;
	//pos = match_no_continuity(calc_continuity_is_faile);
	pos = match_continuity(calc_continuity_is_faile);
	if (std::isnan(pos.x) || std::isnan(pos.y))
	{
		calc_continuity_is_faile = true;
	}

	if (!calc_continuity_is_faile)
	{
		last_pos = pos;
		continuity_retry = 0;

		if (isMatchAllMap)
		{
			isContinuity = false;
			isMatchAllMap = false;
		}
		else
		{
			isContinuity = true;
		}

		is_success_match = true;
	}
	else
	{
		pos = last_pos;
		is_success_match = false;
		continuity_retry++;

		if (continuity_retry >= max_continuity_retry)
		{
			isMatchAllMap = true;
			isContinuity = false;
			continuity_retry = 0;
		}
	}
}

cv::Point2d Tracking::match_continuity(bool& calc_continuity_is_faile)
{
	static cv::Mat img_scene(_mapMat);
	int real_some_map_size_r = DEFAULT_SOME_MAP_SIZE_R;

	cv::Point2d pos_object;

	//cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
	cv::Mat img_object = _miniMapMat;
	cv::Point some_map_center_pos = pos;
	auto someMap_roi = TianLi::Utils::get_some_map_rect(img_scene, some_map_center_pos, DEFAULT_SOME_MAP_SIZE_R);
	cv::Mat someMap = img_scene(someMap_roi).clone();
	if (someMap.empty())
	{
		calc_continuity_is_faile = true;
		return {};
	}

	cv::Mat miniMap = img_object.clone();
	m_matcher->detect_and_compute(miniMap, mini_map_kp);
	mini_map_kp = TianLi::Utils::remove_minimap_fake_keypoint(img_object.size(), _miniMapDiameter * MINIMAP_BORDER_CROP_RATIO, mini_map_kp);

	m_lsh_index->query_and_gather(someMap_roi, map_kp.keypoints, map_kp.descriptors, some_map_kp.keypoints, some_map_kp.descriptors);
#ifdef _CVAT_DEBUG
	// 作图需要_将特征点映射到局部坐标系上，实际发布版本不做映射
	cv::parallel_for_({ 0, static_cast<int>(some_map_kp.keypoints.size()) }, [&](const cv::Range& range) {
		for (int i = range.start; i < range.end; i++)
		{
			some_map_kp.keypoints[i].pt.x -= someMap_roi.x;
			some_map_kp.keypoints[i].pt.y -= someMap_roi.y;
		}
		});
#endif

	// 如果搜索范围内可识别特征点数量少于2，则认为计算失败
	if (some_map_kp.size() <= 2 || mini_map_kp.size() <= 2)
	{
		calc_continuity_is_faile = true;
		return pos_object;
	}

	cv::Point2d p = match_impl(someMap, some_map_kp, img_object, mini_map_kp, calc_continuity_is_faile);
	if (calc_continuity_is_faile)
	{
		return {};
	}

#ifdef _CVAT_DEBUG
	pos_object = cv::Point2d(p.x + some_map_center_pos.x - real_some_map_size_r, p.y + some_map_center_pos.y - real_some_map_size_r);
#else
	pos_object = p;
#endif

	double last_distance = std::sqrt(std::pow(static_cast<double>(pos_object.x) - last_pos.x, 2) +
		std::pow(static_cast<double>(pos_object.y) - last_pos.y, 2));
	if (!isMatchAllMap && last_pos.x != 0 && last_pos.y != 0 && last_distance > 200)
	{
		calc_continuity_is_faile = true;
		return {};
	}

	return pos_object;
}

/// <summary>
/// 匹配 不连续 全局匹配
/// </summary>
/// <param name="calc_is_faile"></param>
/// <returns></returns>
cv::Point2d Tracking::match_no_continuity(bool& calc_is_faile)
{
	//cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
	cv::Mat img_object = _miniMapMat;
	m_matcher->detect_and_compute(img_object, mini_map_kp);
	mini_map_kp = TianLi::Utils::remove_minimap_fake_keypoint(img_object.size(), _miniMapDiameter * MINIMAP_BORDER_CROP_RATIO, mini_map_kp);
	if (mini_map_kp.size() <= 2)
	{
		calc_is_faile = true;
		return {};
	}
	return match_impl(_mapMat, map_kp, img_object, mini_map_kp, calc_is_faile);
}

cv::Point2d Tracking::match_impl(const cv::Mat& img_scene, const IMatcher::KeyMatPoint& keypoint_scene, const cv::Mat& img_object, const IMatcher::KeyMatPoint& keypoint_object, bool& calc_is_faile)
{
	// 没有提取到特征点直接返回，结果无效
	if (keypoint_object.keypoints.size() == 0)
	{
		calc_is_faile = true;
		return {};
	}

	std::vector<cv::Point2f> good_matched_scene;
	std::vector<cv::Point2f> good_matched_object;
	// 匹配特征点
	if (isContinuity)  //连续匹配下图像较小，使用互匹配提高质量
	{
		std::vector<std::vector<cv::DMatch>> KNN_m = m_matcher->knnmatch(keypoint_object, keypoint_scene, 2, true);
		TianLi::Utils::calc_good_matches(img_scene, keypoint_scene.keypoints, img_object, keypoint_object.keypoints, KNN_m, LOWE_RATIO_THRESH_CONTINUITY, good_matched_scene, good_matched_object);
		//auto good_matched_count = good_matched_scene.size();
	}
	else {  //全图匹配较大，优先使用速度更快的lowe
		std::vector<std::vector<cv::DMatch>> KNN_m = m_matcher->knnmatch(keypoint_object, keypoint_scene, 2, false);
		// 绘制关键点
		//cv::Mat match_results;
		//cv::drawMatches(img_object, keypoint_object.keypoints, img_scene, keypoint_scene.keypoints, KNN_m, match_results, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<std::vector<char>>());

		TianLi::Utils::calc_good_matches(img_scene, keypoint_scene.keypoints, img_object, keypoint_object.keypoints, KNN_m, LOWE_RATIO_THRESH, good_matched_scene, good_matched_object);

		//auto good_matched_count = good_matched_scene.size();
	}

	if (good_matched_scene.size() < 6)
	{
		return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
	}

	cv::Mat H, mask;
	H = cv::estimateAffinePartial2D(cv::Mat(good_matched_object), cv::Mat(good_matched_scene), mask, cv::RANSAC);

	int accept_count = cv::countNonZero(mask);

	if (accept_count < 4 || static_cast<double>(accept_count) / good_matched_scene.size() < 0.3)
	{
		//矩阵的置信度不高，使用旧版的筛选算法
		return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
	}
	// 新增几何约束检查
	if (!H.empty() && H.type() == CV_64F) {
		// 提取旋转缩放矩阵部分
		cv::Mat R = H(cv::Rect(0, 0, 2, 2));

		// 使用SVD分解计算旋转角度
		cv::Mat W, U, Vt;
		cv::SVD::compute(R, W, U, Vt);
		cv::Mat R_norm = U * Vt;  // 去除缩放的正交旋转矩阵

		// 计算旋转角度（弧度转角度）
		double angle = std::atan2(R_norm.at<double>(1, 0), R_norm.at<double>(0, 0));
		double angle_deg = std::abs(angle * 180.0 / CV_PI);

		// 计算缩放因子（取两个主方向的均值）
		double scale_x = cv::norm(R.col(0));
		double scale_y = cv::norm(R.col(1));
		double scale = (scale_x + scale_y) / 2.0;

		// 约束条件阈值
		const double MAX_ANGLE = 2.0;    // ±5度
		const double MIN_SCALE = 0.31;    // 最小缩放
		const double MAX_SCALE = 1.3;    // 最大缩放

		if (angle_deg <= MAX_ANGLE && scale >= MIN_SCALE && scale <= MAX_SCALE)
		{
			std::vector<cv::Point2f> out_pt{ cv::Point2f(img_object.cols / 2.0, img_object.rows / 2.0) };
			cv::transform(out_pt, out_pt, H);
			return out_pt[0];
		}
	}
	//不满足约束
	return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
}

cv::Point2d Tracking::cleanAndComputePos_Old(std::vector<cv::Point2f>& good_matched_scene, std::vector<cv::Point2f>& good_matched_object, bool& calc_is_faile)
{
	if (isContinuity)
	{
		// 连续匹配情况下，不使用旧版稀疏算法，以防止陷入局部最优
		calc_is_faile = true;
		return {};
	}

	cv::Point2d all_map_pos{};

	std::vector<double> lisx;
	std::vector<double> lisy;
	lisx.reserve(good_matched_scene.size());
	lisy.reserve(good_matched_scene.size());
	std::transform(good_matched_scene.begin(), good_matched_scene.end(), std::back_inserter(lisx), [](const cv::Point2f& p) {return p.x; });
	std::transform(good_matched_scene.begin(), good_matched_scene.end(), std::back_inserter(lisy), [](const cv::Point2f& p) {return p.y; });

	std::vector<cv::Point2d> list_filter_kp;
	for (int i = 0; i < good_matched_scene.size(); i++)
	{
		list_filter_kp.push_back(cv::Point2d(lisx[i], lisy[i]));
	}
	list_filter_kp = TianLi::Utils::extract_valid(list_filter_kp);

	lisx.clear();
	lisy.clear();
	for (int i = 0; i < list_filter_kp.size(); i++)
	{
		lisx.push_back(list_filter_kp[i].x);
		lisy.push_back(list_filter_kp[i].y);
	}

	// 没有最佳匹配结果直接返回，结果无效
	if (std::min(lisx.size(), lisy.size()) == 0)
	{
		calc_is_faile = true;
		return all_map_pos;
	}
	// 从最佳匹配结果中剔除异常点计算角色位置返回
	if (!TianLi::Utils::SPC(lisx, lisy, all_map_pos))
	{
		calc_is_faile = true;
		return all_map_pos;
	}
	return all_map_pos;
}

cv::Point2d Tracking::getLocalPos()
{
	return pos;
}

bool Tracking::getIsContinuity()
{
	return isContinuity;
}