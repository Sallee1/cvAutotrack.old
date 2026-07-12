#pragma once
#include "match/type/MatchType.h"
#include <memory>

class IMatcher;

namespace TianLi::Genshin::Match
{
	/**
	 * @brief 初始化匹配器（下载资源 + 生成特征点缓存）
	 * @param tile_matcher 用于瓦片特征点生成（无金字塔，单尺度）
	 * @param tracking_matcher 用于追踪匹配（可带金字塔多尺度）
	 * 不依赖游戏句柄，可在启动时尽早调用。
	 */
	void init_matcher(const std::shared_ptr<IMatcher>& tile_matcher, const std::shared_ptr<IMatcher>& tracking_matcher);

	/**
	 * @brief 反初始化匹配器，释放资源
	 */
	void uninit_matcher();

	/**
	 * @brief 查询匹配器是否已完成初始化（线程安全）
	 */
	bool is_matcher_ready();

	/**
	 * @brief 获取角色坐标（依赖 init_matcher 已调用）
	 */
	void get_avatar_position(const GenshinMinimap& genshin_minimap, GenshinAvatarPosition& out_genshin_position, bool is_match_all_map = false);
}