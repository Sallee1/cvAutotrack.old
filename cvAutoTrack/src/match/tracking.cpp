#include "pch.h"
#include "tracking.h"
#include "match/type/MatchType.h"
#include "resources/Resources.h"
#include "resources/KeypointsCache.h"
#include "utils/Utils.h"

void Tracking::setMap(cv::Mat gi_map)
{
	m_mapMat = gi_map;
}

void Tracking::setMiniMap(cv::Mat miniMapMat, float diameter)
{
	m_miniMapMat = miniMapMat;
	//灰度化处理，避免一些算法bug，如FAST
	if(miniMapMat.channels() == 4)
	{
		cv::cvtColor(m_miniMapMat, m_miniMapMat, cv::COLOR_BGRA2GRAY);
	}
	else if (miniMapMat.channels() == 3)
	{
		cv::cvtColor(m_miniMapMat, m_miniMapMat, cv::COLOR_BGR2GRAY);
	}
	if (diameter > 0)
	{
		m_miniMapDiameter = diameter;
	}
	else
	{
		m_miniMapDiameter = static_cast<float>(std::min(miniMapMat.cols, miniMapMat.rows));
	}

	//缩放处理，目前发现反而会破坏特征
	//double scale_ratio = MINIMAP_DIAMETER / m_miniMapDiameter;
	//cv::resize(m_miniMapMat, m_miniMapMat, {}, scale_ratio, scale_ratio, scale_ratio < 0 ? cv::INTER_AREA : cv::INTER_LINEAR);
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher)
{
	if (m_isInit)return true;
	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;
	if(!m_mapMat.empty())
	{
		m_matcher->detect_and_compute(m_mapMat, m_map_kp.keypoints, m_map_kp.descriptors);
		m_matcher->cache_flann_train_descriptors(m_map_kp.descriptors);
	}
	m_isInit = true;
	return true;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher,int cols,int rows, std::vector<cv::KeyPoint>&& gi_map_keypoints, cv::Mat&& gi_map_descriptors)
{
	if (m_isInit)return true;
	m_map_kp.keypoints = std::move(gi_map_keypoints);
	m_map_kp.descriptors = std::move(gi_map_descriptors);
	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;
	m_matcher->cache_flann_train_descriptors(m_map_kp.descriptors);

	auto cell_size = Resources::getInstance().lsh_cell_size;
	m_lsh_index = std::make_unique<KeypointGridLSH>();
	m_lsh_index->build(m_map_kp.keypoints, cv::Rect2i(0, 0, cols, rows), cv::Size2i(cell_size, cell_size));

	m_isInit = true;
	return true;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher, MapKeypointCache&& map_keypoints_cache)
{
	if (m_isInit) return false;
	m_lsh_index = std::make_unique<KeypointGridLSH>();
	m_lsh_index->fromCache(map_keypoints_cache);
	m_map_kp.keypoints = std::move(map_keypoints_cache.keypoints);
	m_map_kp.descriptors = std::move(map_keypoints_cache.descriptors);

	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;
	m_matcher->cache_flann_train_descriptors(m_map_kp.descriptors);
	m_isInit = true;
	return true;
}

void Tracking::UnInit()
{
	if (!m_isInit)return;
	m_mapMat = cv::Mat();
	m_map_kp.keypoints.clear();
	m_map_kp.descriptors.release();
	m_isInit = false;
}

void Tracking::match()
{
	bool calc_is_faile = false;
	m_is_success_match = false;

	// 非连续匹配，匹配整个大地图
	if (m_isMatchAllMap)
	{
		m_pos = match_no_continuity(calc_is_faile);
		if (std::isnan(m_pos.x) || std::isnan(m_pos.y))
		{
			calc_is_faile = true;
		}

		// 没有有效结果，结束
		if (calc_is_faile)
		{
			m_pos = m_last_pos;
			m_is_success_match = false;
			return;
		}
		m_continuity_retry = m_max_continuity_retry - 1;		//全局检测后只局部检测一次
	}

	// 尝试连续匹配，匹配角色附近小范围区域
	bool calc_continuity_is_faile = false;
	//pos = match_no_continuity(calc_continuity_is_faile);
	m_pos = match_continuity(calc_continuity_is_faile);
	if (std::isnan(m_pos.x) || std::isnan(m_pos.y))
	{
		calc_continuity_is_faile = true;
	}

	if (!calc_continuity_is_faile)
	{
		m_last_pos = m_pos;
		m_continuity_retry = 0;

		if (m_isMatchAllMap)
		{
			m_isContinuity = false;
			m_isMatchAllMap = false;
		}
		else
		{
			m_isContinuity = true;
		}

		m_is_success_match = true;
	}
	else
	{
		m_pos = m_last_pos;
		m_is_success_match = false;
		m_continuity_retry++;

		if (m_continuity_retry >= m_max_continuity_retry)
		{
			m_isMatchAllMap = true;
			m_isContinuity = false;
			m_continuity_retry = 0;
		}
	}
}

cv::Point2d Tracking::match_continuity(bool& calc_continuity_is_faile)
{
	static cv::Mat img_scene(m_mapMat);
	int real_some_map_size_r = DEFAULT_SOME_MAP_SIZE_R;

	cv::Point2d pos_object;

	//cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
	cv::Mat img_object = m_miniMapMat;
	cv::Point some_map_center_pos = m_pos;
	auto keypoint_roi = TianLi::Utils::get_rect_by_center_r(some_map_center_pos, DEFAULT_SOME_MAP_SIZE_R);

	cv::Mat someMap = cv::Mat();
#ifdef _CVAT_DEBUG
	if(!img_scene.empty())
	{
		// 考虑到界外特征点的存在，目前采取的方式是，对于界外特征点，使用空图像填充确保可作图
		cv::Rect2i someMap_roi = keypoint_roi & cv::Rect(0, 0, img_scene.cols, img_scene.rows);
		if (someMap_roi.width == 0 || someMap_roi.height == 0)
		{
			someMap = cv::Mat::zeros(cv::Size(real_some_map_size_r * 2, real_some_map_size_r * 2), img_scene.type());
		}
		else {
			someMap = img_scene(someMap_roi).clone();
		}
	}
#endif

	cv::Mat miniMap = img_object.clone();
	IMatcher::KeyMatPoint some_map_kp;
	IMatcher::KeyMatPoint mini_map_kp;
	
	m_matcher->detect_and_compute(miniMap, mini_map_kp);
	mini_map_kp = TianLi::Utils::remove_minimap_fake_keypoint(img_object.size(), m_miniMapDiameter * MINIMAP_BORDER_CROP_RATIO, mini_map_kp);

	m_lsh_index->query_and_gather(keypoint_roi, m_map_kp.keypoints, m_map_kp.descriptors, some_map_kp.keypoints, some_map_kp.descriptors);

#ifdef _CVAT_DEBUG
    //调试旁路代码
    if (!img_scene.empty()) {
        // 作图需要_将特征点映射到局部坐标系上，实际发布版本不做映射
        cv::parallel_for_({ 0, static_cast<int>(some_map_kp.keypoints.size()) }, [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++)
            {
                some_map_kp.keypoints[i].pt.x -= keypoint_roi.x;
                some_map_kp.keypoints[i].pt.y -= keypoint_roi.y;
            }
            });
    }
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
    if (!img_scene.empty()) {
        pos_object = cv::Point2d(p.x + some_map_center_pos.x - real_some_map_size_r, p.y + some_map_center_pos.y - real_some_map_size_r);
    }
    else {
#else
    {
#endif
        pos_object = p;
    }

	double last_distance = std::sqrt(std::pow(static_cast<double>(pos_object.x) - m_last_pos.x, 2) +
		std::pow(static_cast<double>(pos_object.y) - m_last_pos.y, 2));
	if (!m_isMatchAllMap && m_last_pos.x != 0 && m_last_pos.y != 0 && last_distance > DEFAULT_SOME_MAP_SIZE_R)
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
	cv::Mat img_object = m_miniMapMat;
	IMatcher::KeyMatPoint mini_map_kp;
	m_matcher->detect_and_compute(img_object, mini_map_kp);
	mini_map_kp = TianLi::Utils::remove_minimap_fake_keypoint(img_object.size(), m_miniMapDiameter * MINIMAP_BORDER_CROP_RATIO, mini_map_kp);
	//Lowe测试
	std::vector<cv::DMatch> good_matches;
	std::vector<cv::Point2f> good_matched_scene;
	std::vector<cv::Point2f> good_matched_object;
	std::vector<std::vector<cv::DMatch>> KNN_m = m_matcher->flann_knnmatch(mini_map_kp, 2);
	TianLi::Utils::lowe_test(KNN_m, LOWE_RATIO_THRESH, good_matches);
	TianLi::Utils::dmatch2cvPoints(m_map_kp.keypoints, mini_map_kp.keypoints, good_matches, good_matched_scene, good_matched_object);

#ifdef _CVAT_DEBUG
	if (!m_mapMat.empty())
	{
		TianLi::Utils::draw_good_matches(m_mapMat, m_map_kp.keypoints, img_object, mini_map_kp.keypoints, good_matches);
	}
#endif


	//尝试探索局部点
	const cv::Size2i rect_size = cv::Size2i{ DEFAULT_SOME_MAP_SIZE_R*2,DEFAULT_SOME_MAP_SIZE_R * 2 };
	std::vector<cv::Rect2i> rects_maybe = TianLi::Utils::getRectsByPoints(good_matched_scene, rect_size);
	for (auto& rect : rects_maybe)
	{
		IMatcher::KeyMatPoint some_map_kp;
		m_lsh_index->query_and_gather(rect, m_map_kp.keypoints, m_map_kp.descriptors, some_map_kp.keypoints, some_map_kp.descriptors);
		
		cv::Mat some_map;
#ifdef _CVAT_DEBUG
        if (!m_mapMat.empty())
        {
            some_map = m_mapMat(rect);
            // 作图需要_将特征点映射到局部坐标系上，实际发布版本不做映射
            cv::parallel_for_({ 0, static_cast<int>(some_map_kp.keypoints.size()) }, [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; i++)
                {
                    some_map_kp.keypoints[i].pt.x -= rect.x;
                    some_map_kp.keypoints[i].pt.y -= rect.y;
                }
                });
        }
#endif

		cv::Point2d out_pt = match_impl(some_map, some_map_kp, img_object, mini_map_kp, calc_is_faile);


#ifdef _CVAT_DEBUG
        if (!m_mapMat.empty())
        {
            out_pt = cv::Point2d(out_pt.x + rect.x, out_pt.y + rect.y);
        }
#endif

		if(!calc_is_faile)
		{
			return out_pt;
		}
		calc_is_faile = false;
	}
	calc_is_faile = true;
	return {};
}

cv::Point2d Tracking::match_impl(const cv::Mat& img_scene, const IMatcher::KeyMatPoint& keypoint_scene, const cv::Mat& img_object, const IMatcher::KeyMatPoint& keypoint_object, bool& calc_is_faile)
{
	// 没有提取到特征点直接返回，结果无效
	if (keypoint_object.keypoints.size() == 0)
	{
		calc_is_faile = true;
		return {};
	}

	std::vector<cv::DMatch> good_matches;
	std::vector<cv::Point2f> good_matched_scene;
	std::vector<cv::Point2f> good_matched_object;
	std::vector<std::vector<cv::DMatch>> KNN_m = m_matcher->bf_knnmatch(keypoint_object, keypoint_scene, 2);
	TianLi::Utils::lowe_test(KNN_m, LOWE_RATIO_THRESH_CONTINUITY, good_matches);

	TianLi::Utils::dmatch2cvPoints(keypoint_scene.keypoints, keypoint_object.keypoints, good_matches, good_matched_scene, good_matched_object);
#ifdef _CVAT_DEBUG
	if(!img_scene.empty())
	{
		TianLi::Utils::draw_good_matches(img_scene, keypoint_scene.keypoints, img_object, keypoint_object.keypoints, good_matches);
	}
#endif

	if (good_matched_scene.size() < 6)
	{
		calc_is_faile = true; return {};
	}

	cv::Mat H, mask;
	H = cv::estimateAffinePartial2D(cv::Mat(good_matched_object), cv::Mat(good_matched_scene), mask, cv::RANSAC);

	int accept_count = cv::countNonZero(mask);

	if (accept_count < 4 || static_cast<double>(accept_count) / good_matched_scene.size() < 0.3)
	{
		//矩阵的置信度不高，使用旧版的筛选算法
		calc_is_faile = true; return {};
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
			std::vector<cv::Point2f> out_pt{ cv::Point2f(img_object.cols / 2.0f, img_object.rows / 2.0f) };
			cv::transform(out_pt, out_pt, H);
			return out_pt[0];
		}
	}
	//不满足约束
	calc_is_faile = true; return {};
}

[[deprecated]] cv::Point2d Tracking::cleanAndComputePos_Old(std::vector<cv::Point2f>& good_matched_scene, bool& calc_is_faile)
{
	if (m_isContinuity)
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

	list_filter_kp = TianLi::Utils::std_mean_filter(list_filter_kp);
	//list_filter_kp = TianLi::Utils::max_near_fliter(list_filter_kp, _miniMapDiameter * MINIMAP_BORDER_CROP_RATIO * 3 / 2);

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
	return m_pos;
}

bool Tracking::getIsContinuity()
{
	return m_isContinuity;
}