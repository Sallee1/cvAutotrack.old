#include "pch.h"
#include "genshin.match.position.h"

#include "resources/Resources.h"
#include "resources/KeypointsCache.h"
#include "Match/tracking.h"
#include "filter/kalman/Kalman.h"
#include "match/IMatcher.h"

namespace {
	Tracking g_surf_match;
	std::atomic<bool> g_is_init{false};
}

namespace TianLi::Genshin::Match
{
	void init_matcher(const std::shared_ptr<IMatcher>& tile_matcher, const std::shared_ptr<IMatcher>& tracking_matcher)
	{
		if (g_is_init.load()) return;

		// install 包含网络下载 + metadata 加载，在后台线程中执行
		Resources::getInstance().install();

		// 瓦片特征点生成用无金字塔 matcher
		MapKeypointCache map_keypoints_cache = get_map_keypoint(tile_matcher);
		// 追踪匹配用带金字塔的 matcher
		if (!g_surf_match.Init(tracking_matcher, std::move(map_keypoints_cache)))
		{
			// 所有缓存生成和回退均失败，放弃此次初始化
			MessageBox(NULL,
				L"初始化位置追踪失败！\n"
				L"热更新下载失败且本地无可用的特征点缓存。\n"
				L"请检查网络连接，或尝试重新启动程序。\n"
				L"更多信息请查看日志文件。",
				L"严重错误", MB_OK | MB_ICONERROR);
			Resources::getInstance().release();
			return;
		}
#ifdef _CVAT_DEBUG
		g_surf_match.setMap(Resources::getInstance().DebugMapTemplate);
#else
		// 正式发布版本，释放图像，因为目前已经可以实现无图匹配
		Resources::getInstance().release();
#endif
		g_is_init.store(true);
	}

	void uninit_matcher()
	{
		if (!g_is_init.load()) return;
		g_surf_match.UnInit();
		Resources::getInstance().release();
		g_is_init.store(false);
	}

	bool is_matcher_ready()
	{
		return g_is_init.load();
	}
}

cv::Mat to_color(cv::Mat& img_object)
{
	cv::Mat color_mat;
	int s_len = static_cast<int>((img_object.cols + img_object.rows) * 0.25 * 0.8);
	cv::Mat roi_tl = img_object(cv::Rect(0, 0, s_len, s_len));
	cv::Mat roi_tr = img_object(cv::Rect(img_object.cols - s_len, 0, s_len, s_len));
	cv::Mat roi_bl = img_object(cv::Rect(0, img_object.rows - s_len, s_len, s_len));
	cv::Mat roi_br = img_object(cv::Rect(img_object.cols - s_len, img_object.rows - s_len, s_len, s_len));

	cv::Mat roi_tl_color;
	cv::Mat roi_tr_color;
	cv::Mat roi_bl_color;
	cv::Mat roi_br_color;

	cv::resize(roi_tl, roi_tl_color, cv::Size(3, 3), cv::INTER_AREA);
	cv::resize(roi_tr, roi_tr_color, cv::Size(3, 3), cv::INTER_AREA);
	cv::resize(roi_bl, roi_bl_color, cv::Size(3, 3), cv::INTER_AREA);
	cv::resize(roi_br, roi_br_color, cv::Size(3, 3), cv::INTER_AREA);

	cv::Mat roi = cv::Mat::zeros(6, 6, img_object.type());
	roi_tl_color.copyTo(roi(cv::Rect(0, 0, 3, 3)));
	roi_tr_color.copyTo(roi(cv::Rect(3, 0, 3, 3)));
	roi_bl_color.copyTo(roi(cv::Rect(0, 3, 3, 3)));
	roi_br_color.copyTo(roi(cv::Rect(3, 3, 3, 3)));

	cv::Mat roi_color;
	cv::resize(roi, roi_color, cv::Size(1, 1), cv::INTER_AREA);

	roi_color.at<cv::Vec4b>(0, 0)[3] = 255;

	cv::resize(img_object, color_mat, cv::Size(5, 5), cv::INTER_AREA);
	return roi_color;
}
// 初步定位：根据颜色确定角色在大地图的哪个方位
cv::Point match_find_direction_in_all(cv::Mat& _mapMat, cv::Mat& _MiniMapMat)
{
	static cv::Mat color_map;
	static bool is_first = true;
	if (is_first)
	{
		cv::resize(_mapMat, color_map, cv::Size(), 0.01, 0.01, cv::INTER_CUBIC);
		is_first = false;
	}

	int crop_border = static_cast<int>((_MiniMapMat.rows + _MiniMapMat.cols) * 0.5 * 0.15);
	cv::Mat img_object(_MiniMapMat(cv::Rect(crop_border, crop_border, _MiniMapMat.cols - crop_border * 2, _MiniMapMat.rows - crop_border * 2)));
	// 对小地图进行颜色提取
	cv::Mat color = to_color(img_object);
	// 模板匹配
	cv::Mat result;
	cv::matchTemplate(color_map, color, result, cv::TM_CCOEFF_NORMED);
	// 找到最大值和最小值的位置
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	// 计算角色在大地图的位置
	cv::Point pos = cv::Point(maxLoc.x * 100 + 50, maxLoc.y * 100 + 50);
	return pos;
}
// 确定区块：根据初步定位的结果再遍历该方位的区块，确定所在区块
cv::Point match_find_block_in_direction(cv::Mat& _mapMat, cv::Mat& _MiniMapMat, cv::Point pos_first_match)
{
	// 0 18 22 38
	const static cv::Rect yellow_rect = cv::Rect(0, 18, 22, 38);
	if (yellow_rect.contains(pos_first_match))
	{
		return cv::Point(-1, 0);
	}
	UNREFERENCED_PARAMETER(_mapMat);
	UNREFERENCED_PARAMETER(_MiniMapMat);
	return cv::Point(0, 0);
}
// 确定位置：根据所在区块的结果精确匹配角色位置
cv::Point2d match_yellow_block(cv::Mat& _mapMat, cv::Mat& _MiniMapMat)
{
	cv::Point pos_first_match = match_find_direction_in_all(_mapMat, _MiniMapMat);
	cv::Point pos_block_match = match_find_block_in_direction(_mapMat, _MiniMapMat, pos_first_match);
	cv::Point2d pos_continuity_no;
	bool calc_is_faile = false;
	//cv::Point2d pos = match_find_pos_in_block(_mapMat, _MiniMapMat, pos_block_match, pos_continuity_no, calc_is_faile);
	if (calc_is_faile)
	{
		return cv::Point2d(-1, -1);
	}
	return cv::Point2d(0, 0);
}
// 确定位置：根据所在区块的结果精确匹配角色位置
cv::Point2d match_find_position_in_block(cv::Point pos_second_match, bool& calc_is_faile)
{
	if (pos_second_match.x == -1)
	{
		return cv::Point2d(0, 0);
	}
	else
	{
		UNREFERENCED_PARAMETER(calc_is_faile);
		//return match_no_continuity_1st(calc_is_faile);
		return cv::Point2d(0, 0);
	}
}
/// <summary>
/// 非连续匹配，从大地图中确定角色位置 v2.0
/// 根据某小地图整体颜色判断角色的大致位置，然后再根据大致位置进行精确匹配
/// </summary>
/// <param name="calc_is_faile">匹配结果是否有效</param>
/// <returns></returns>
cv::Point2d match_no_continuity_2nd(bool& calc_is_faile)
{
	cv::Point pos_first_match;
	cv::Point pos_second_match;
	cv::Point2d pos_continuity_no;
	// 初步定位：根据颜色确定角色在大地图的哪个方位
	//pos_first_match = match_find_direction_in_all(_mapMat, _miniMapMat);
	// 确定区块：根据初步定位的结果再遍历该方位的区块，确定所在区块
	//pos_second_match = match_find_block_in_direction(_mapMat, _miniMapMat, pos_first_match);
	// 确定位置：根据所在区块的结果精确匹配角色位置
	pos_continuity_no = match_find_position_in_block(pos_second_match, calc_is_faile);
	// 返回结果
	return pos_continuity_no;
}

void TianLi::Genshin::Match::get_avatar_position(const GenshinMinimap& genshin_minimap, GenshinAvatarPosition& out_genshin_position, bool is_match_all_map)
{
	if (!g_is_init.load())
	{
		return;
	}

	if (genshin_minimap.img_minimap.empty())
	{
		return;
	}

	g_surf_match.setMiniMap(genshin_minimap);

    if (is_match_all_map)
    {
        g_surf_match.setMatchAllMapNext();
    }
	g_surf_match.match();

	out_genshin_position.position = g_surf_match.getLocalPos();
	out_genshin_position.config.is_continuity = g_surf_match.m_isContinuity;

	if (out_genshin_position.config.is_use_filter)
	{
		cv::Point2d pos = out_genshin_position.position;
		cv::Point2d filt_pos;
		if (out_genshin_position.config.is_continuity == false)
		{
			filt_pos = out_genshin_position.config.pos_filter->re_init_filterting(pos);
		}
		else
		{
			filt_pos = out_genshin_position.config.pos_filter->filterting(pos);
		}
		out_genshin_position.position = filt_pos;
	}

	if (g_surf_match.m_is_success_match)
	{
		out_genshin_position.config.img_last_match_minimap = genshin_minimap.img_minimap.clone();
		out_genshin_position.config.is_exist_last_match_minimap = true;
	}
	else
	{
		out_genshin_position.config.is_exist_last_match_minimap = false;
	}
}