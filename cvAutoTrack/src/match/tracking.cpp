#include "pch.h"
#include "tracking.h"
#include "match/type/MatchType.h"
#include "match/LinearSolve.h"
#include "resources/Resources.h"
#include "resources/KeypointsCache.h"
#include "utils/Utils.h"

void Tracking::setMap(cv::Mat gi_map)
{
	m_mapMat = gi_map;
}

void Tracking::setMiniMap(const GenshinMinimap& minimap)
{
	// img_minimap_padding → 带遮罩+padding，用于特征点匹配
	m_miniMapMat = minimap.img_minimap_padding.clone();
	// img_minimap → 原始裁剪（无遮罩无padding），用于相位相关
	m_miniMapCenter = minimap.img_viewer.clone();
	m_miniMapDiameter = minimap.minimap_diameter;

	//灰度化处理，避免一些算法bug，如FAST
	auto toGray = [](cv::Mat& mat)
	{
		if (mat.channels() == 4)
			cv::cvtColor(mat, mat, cv::COLOR_BGRA2GRAY);
		else if (mat.channels() == 3)
			cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	};
	toGray(m_miniMapMat);
	toGray(m_miniMapCenter);
}

void Tracking::setMatchAllMapNext() {
    m_isMatchAllMap = true;
}

/**
 * @brief 强制下一次不使用惯性导航
 */
void Tracking::setNoInertialNavigatorNext() {
    m_isNoInertialNavigator = true;
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
		m_matcher->cache_train_descriptors(m_map_kp.descriptors);
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
	m_matcher->cache_train_descriptors(m_map_kp.descriptors);

	auto cell_size = Resources::getInstance().lsh_cell_size;
	m_lsh_index = std::make_unique<KeypointGridLSH>();
	m_lsh_index->build(m_map_kp.keypoints, cv::Rect2i(0, 0, cols, rows), cv::Size2i(cell_size, cell_size));

	m_isInit = true;
	return true;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher, MapKeypointCache&& map_keypoints_cache)
{
	if (m_isInit) return false;

	// 拒绝空数据，防止后续匹配操作段错误
	if (map_keypoints_cache.keypoints.empty() || map_keypoints_cache.descriptors.empty())
	{
		return false;
	}

	m_lsh_index = std::make_unique<KeypointGridLSH>();
	m_lsh_index->fromCache(map_keypoints_cache);
	m_map_kp.keypoints = std::move(map_keypoints_cache.keypoints);
	m_map_kp.descriptors = std::move(map_keypoints_cache.descriptors);

	if (matcher == nullptr)
	{
		return false;
	}
	m_matcher = matcher;
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
	do {
		// ============================================================
		// 模式 A：全局匹配（首次启动 / 传送后重新定位）
		// ============================================================
		if (m_isMatchAllMap)
		{
			m_isContinuity = false;

			m_pos = match_no_continuity(calc_is_faile);
			if (std::isnan(m_pos.x) || std::isnan(m_pos.y))
				calc_is_faile = true;

			if (calc_is_faile)
			{
				m_pos = m_last_pos;
				m_is_success_match = false;
				m_isMatchAllMap = true;
#ifdef _CVAT_DEBUG_LOG
                printf("[DEBUG] 全局匹配失败\n");
#endif
				break;
			}

			m_last_pos = m_pos;
			m_isMatchAllMap = false;
			m_is_success_match = true;
			m_inertial.reset();
#ifdef _CVAT_DEBUG_LOG
                printf("[DEBUG] 全局匹配成功\n");
#endif
			break;
		}

		// ============================================================
		// 模式 B：连续追踪（主模式 = 局部特征匹配）
		// ============================================================
		m_isContinuity = true;

		// B1. 首选：局部特征点匹配 → 亚像素精度，极少假阳性
		bool continuity_failed = false;
		m_pos = match_continuity(continuity_failed);
		if (std::isnan(m_pos.x) || std::isnan(m_pos.y))
			continuity_failed = true;

		if (!continuity_failed)
		{
			m_last_pos = m_pos;
			m_isMatchAllMap = false;
			m_is_success_match = true;
			m_inertial.reset();
#ifdef _CVAT_DEBUG_LOG
            printf("[DEBUG] 局部匹配成功\n");
#endif
			break;
		}

		m_pos = m_last_pos;
		m_is_success_match = false;

		if (m_last_pos.x == 0 && m_last_pos.y == 0)
			break;  // 无有效历史位置，无法惯性导航
		if (m_miniMapCenter.empty())
			break;

        // 不使用惯性导航
        if (m_isNoInertialNavigator)
        {
            m_isMatchAllMap = true;
            m_isNoInertialNavigator = false;
            break;
        }

		// B2. 特征匹配失败 → 惯性导航补位（帧间相位相关）
		double peak = 0.0;
		cv::Point2d diff_delta = {NAN, NAN};
		double pixel_dist = 0.0;

		if (!m_inertial.hasLastFrame()
			|| m_inertial.last_frame.size() != m_miniMapCenter.size())
		{
			// 首帧或尺寸不一致 → 无法做帧间相关，跳过本帧
			peak = 0.0;
		}
		else
		{
			diff_delta = DiffMatch::phaseCorrelate(
				m_inertial.last_frame, m_miniMapCenter, peak);
		}

		// B2a. 一票否决：峰值突降 → 场景剧变（传送） → 切全局匹配
		if (m_inertial.needsVeto(peak))
		{
			m_isMatchAllMap = true;
			m_isContinuity = false;
			m_inertial.reset();
			break;
		}

		// B2b. 相位相关可靠 → 积分位移更新位置
		double displacement = 0.0;  // 坐标空间位移
		if (peak >= DiffMatch::CONFIDENCE_THRESHOLD
			&& !std::isnan(diff_delta.x) && !std::isnan(diff_delta.y))
		{
			pixel_dist = std::sqrt(
				diff_delta.x * diff_delta.x +
				diff_delta.y * diff_delta.y);

			double scale = m_tracking_scale;
			double new_x = m_last_pos.x + diff_delta.x * scale;
			double new_y = m_last_pos.y + diff_delta.y * scale;

			double coord_dist = pixel_dist * scale;
			if (coord_dist < DEFAULT_SOME_MAP_SIZE_R)
			{
				m_pos.x = new_x;
				m_pos.y = new_y;
				m_last_pos = m_pos;
				m_is_success_match = true;
				displacement = coord_dist;
#ifdef _CVAT_DEBUG_LOG
                printf("[DEBUG] 惯性导航成功\n");
#endif
			}
		}

		m_inertial.record(peak, displacement, pixel_dist);

		// 进入惯性模式首帧 → 缓存关键帧，供 B2c 大步校正使用
		if (m_is_success_match && !m_inertial.hasKeyframe())
		{
			m_inertial.captureKeyframe(m_miniMapCenter, m_pos);
		}

		// B2c. 漂移超阈值 → 大步关键帧校正
		if (m_inertial.needsCorrection())
		{
			m_inertial.markCorrectionAttempted();

			if (m_inertial.hasKeyframe())
			{
				double kf_peak = 0.0;
				cv::Point2d kf_delta = DiffMatch::phaseCorrelate(
					m_inertial.keyframe, m_miniMapCenter, kf_peak);

				if (kf_peak >= InertialNavigator::KEYFRAME_PEAK_THRESHOLD
					&& !std::isnan(kf_delta.x) && !std::isnan(kf_delta.y))
				{
					double scale = m_tracking_scale;
					double corrected_x = m_inertial.keyframe_pos.x + kf_delta.x * scale;
					double corrected_y = m_inertial.keyframe_pos.y + kf_delta.y * scale;

					double kf_distance = std::sqrt(
						kf_delta.x * kf_delta.x +
						kf_delta.y * kf_delta.y) * scale;

					if (kf_distance < DEFAULT_SOME_MAP_SIZE_R)
					{
						m_pos.x = corrected_x;
						m_pos.y = corrected_y;
						m_last_pos = m_pos;
						m_is_success_match = true;
						// 刷新关键帧
						m_inertial.captureKeyframe(m_miniMapCenter, m_pos);
#ifdef _CVAT_DEBUG_LOG
						printf("[DEBUG] 惯性导航校准成功\n");
#endif
						break;
					}
				}
			}

			// 关键帧校正失败 → 切全局匹配
			m_isMatchAllMap = true;
			m_isContinuity = false;
			m_inertial.reset();
#ifdef _CVAT_DEBUG_LOG
			printf("[DEBUG] 关键帧校正失败，切全局匹配\n");
#endif
		}

		// B2d. 惯性超限 → 终止惯性，切全局匹配（避免累积误差过大）
		if (m_inertial.consecutive_frames >= InertialNavigator::MAX_INERTIAL_FRAMES)
		{
			m_isMatchAllMap = true;
			m_isContinuity = false;
			m_inertial.reset();
#ifdef _CVAT_DEBUG_LOG
			printf("[DEBUG] 惯性导航超限，切全局匹配\n");
#endif
			break;
		}

	} while (false);

	// 每帧更新惯性导航的上一帧，保持帧间相位相关随时可用
	m_inertial.updateLastFrame(m_miniMapCenter);
}

cv::Point2d Tracking::match_continuity(bool& calc_continuity_is_faile)
{
	const cv::Mat& img_scene = m_mapMat;
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
	
	m_matcher->detect_and_compute_ex(miniMap, mini_map_kp);
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

	cv::Point2d p = match_impl(some_map_kp, keypoint_roi, mini_map_kp , calc_continuity_is_faile);
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
	const cv::Mat &img_scene = m_mapMat;
	const cv::Mat &img_object = m_miniMapMat;
	IMatcher::KeyMatPoint mini_map_kp;
	m_matcher->detect_and_compute_ex(img_object, mini_map_kp);
	mini_map_kp = TianLi::Utils::remove_minimap_fake_keypoint(img_object.size(), m_miniMapDiameter * MINIMAP_BORDER_CROP_RATIO, mini_map_kp);
	//Lowe测试
	std::vector<cv::DMatch> good_matches;
	std::vector<cv::Point2f> good_matched_scene;
	std::vector<cv::Point2f> good_matched_object;
	std::vector<std::vector<cv::DMatch>> KNN_m = m_matcher->indexed_knnmatch(mini_map_kp, 2);
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
        const cv::Rect2i mapMatBorder = cv::Rect(0, 0, m_mapMat.cols, m_mapMat.rows);
        cv::Rect2i rect_clamp = rect & mapMatBorder;
        if (!m_mapMat.empty() && rect_clamp.width > 0 && rect_clamp.height > 0)
        {
            some_map = m_mapMat(rect_clamp);
        }
        // 作图需要_将特征点映射到局部坐标系上，实际发布版本不做映射
        cv::parallel_for_({ 0, static_cast<int>(some_map_kp.keypoints.size()) }, [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++)
            {
                some_map_kp.keypoints[i].pt.x -= rect_clamp.x;
                some_map_kp.keypoints[i].pt.y -= rect_clamp.y;
            }
        });
#endif

		cv::Point2d out_pt = match_impl(some_map_kp, rect, mini_map_kp, calc_is_faile);


#ifdef _CVAT_DEBUG
        if (!m_mapMat.empty())
        {
            out_pt = cv::Point2d(out_pt.x + rect_clamp.x, out_pt.y + rect_clamp.y);
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

cv::Point2d Tracking::match_impl(const IMatcher::KeyMatPoint& keypoint_scene, const cv::Rect2i& keypoint_roi, const IMatcher::KeyMatPoint& keypoint_object, bool& calc_is_faile)
{
	cv::Mat img_scene;
	if(!m_mapMat.empty())
	{
		cv::Rect2i mapMatRoi{cv::Point2i{}, m_mapMat.size()};
		cv::Rect2i roi = mapMatRoi & keypoint_roi;
		if(roi.size() != cv::Size2i{0,0})
		{
			img_scene = m_mapMat(roi);
		}
	}
	const cv::Mat &img_object = m_miniMapMat;

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

	const int N = static_cast<int>(good_matched_scene.size());

	std::vector<cv::Point2d> src_pts(N), dst_pts(N);
	for (int i = 0; i < N; i++) {
		src_pts[i] = cv::Point2d(good_matched_object[i]);
		dst_pts[i] = cv::Point2d(good_matched_scene[i]);
	}

	// 线性求解器估计约束
	cv::Mat H = LinearSolve::estimateScaleTranslation(src_pts, dst_pts);
	if (H.empty()) {
		calc_is_faile = true; return {};
	}

	double best_s, best_dx, best_dy;
	LinearSolve::decomposeST(H, best_s, best_dx, best_dy);

	updateTrackingScale(best_s);

	// 小地图中心点映射到大地图坐标
	double cx = static_cast<double>(img_object.cols) / 2.0;
	double cy = static_cast<double>(img_object.rows) / 2.0;
	return cv::Point2d(cx * best_s + best_dx, cy * best_s + best_dy);
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