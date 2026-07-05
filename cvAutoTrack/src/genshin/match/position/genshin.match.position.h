#pragma once
#include "match/type/MatchType.h"
#include <memory>

class IMatcher;

namespace TianLi::Genshin::Match
{
	/**
	 * @brief 初始化匹配器（下载资源 + 生成特征点缓存）
	 * 不依赖游戏句柄，可在启动时尽早调用。
	 */
	void init_matcher(const std::shared_ptr<IMatcher>& matcher);

	/**
	 * @brief 反初始化匹配器，释放资源
	 */
	void uninit_matcher();

	/**
	 * @brief 获取角色坐标（依赖 init_matcher 已调用）
	 */
	void get_avatar_position(const GenshinMinimap& genshin_minimap, GenshinAvatarPosition& out_genshin_position);
}