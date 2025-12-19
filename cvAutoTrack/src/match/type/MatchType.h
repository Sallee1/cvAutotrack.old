#pragma once
class Capture;
class Filter;

#include "frame/frame.include.h"
#include <match/IMatcher.h>

enum GenshinWindowClass
{
	None,
	Unity,
	Obs,
	Qt,
};

const std::vector<std::pair<std::wstring, GenshinWindowClass>> GenshinProcessNameList =
{
	{L"YuanShen.exe",GenshinWindowClass::Unity},
	{L"GenshinImpact.exe", GenshinWindowClass::Unity} ,
	//{L"Genshin Impact Cloud Game.exe", GenshinWindowClass::Qt},
};

struct GenshinHandleConfig
{
	bool is_auto_find_genshin = true;
	bool is_force_used_no_alpha = false;
	HWND genshin_handle = nullptr;
	std::shared_ptr<tianli::frame::frame_source> frame_source;

	std::vector<std::pair<std::wstring, GenshinWindowClass>> genshin_process_list = GenshinProcessNameList;
};
// 用于匹配原神窗口的句柄相关变量
struct GenshinHandle
{
	bool is_exist = false;
	HWND handle;
	RECT rect;
	RECT rect_client;
	float scale;
	bool is_exist_title_bar = true;
	cv::Size size_frame;
	GenshinHandleConfig config;
};
struct GenshinScreenConfig
{
	bool is_used_alpha = true;
	bool is_controller_mode = false;
	bool is_search_mode = true;
	float controller_ui_scale = 0.83333f;
	int icon_size = 54;    //键鼠模式下图标尺寸
	int icon_size_ctrl = 45;    //手柄模式下图标尺寸
};
// 用于获取原神画面的相关变量
struct GenshinScreen
{
	cv::Rect rect_client;
	std::chrono::system_clock::time_point last_time = std::chrono::system_clock::now();
	cv::Mat img_screen;

	struct Imgs {
		cv::Mat icon_sight_maybe;
		cv::Mat icon_sight;
		cv::Mat minimap_maybe;
		cv::Mat minimap;
		cv::Mat avatar_maybe;
		cv::Mat avatar;
		cv::Mat uid_maybe;
		cv::Mat uid;
	}imgs;
	struct Rects {
		cv::Rect icon_sight_maybe;
		cv::Rect icon_sight;
		cv::Rect minimap_maybe;
		cv::Rect minimap;
		cv::Rect avatar_maybe;
		cv::Rect avatar;
		cv::Rect uid_maybe;
		cv::Rect uid;
	}rects;

	GenshinScreenConfig config;
};

struct GenshinIconSightConfig
{
	bool is_need_find = true;
	float icon_sight_threshold_low = 0.90f;   //低亮度下的匹配阈值
	float icon_sight_threshold_high = 0.95f;    //高亮度下的匹配阈值
	float ratio = 1.05f;            //容许的模板图像尺寸比
	float min_distance = 0.005f;    //形状的最大差异
	float tplmatch_max_diff = 0.05f;   //模板匹配允许的最大差异
	int min_size = 35;  //容许模板图像的最小尺寸
	int max_size = 51;  //容许模板图像的最大尺寸
	int ctrl_size = 44; //区分控制器的尺寸
};

struct GenshinIconSight
{
	bool is_visial = false;
	bool is_ctrl_mode = false;
	cv::Rect rect_Icon_sight;
	GenshinIconSightConfig config;
};

struct GenshinMinimapConfig {
	cv::Size minimap_size = { 218,218 };
};

struct GenshinMinimap
{
	bool is_init_finish = false;
	bool is_run_init_start = false;
	bool is_run_uninit_start = false;
	bool is_cailb = false;
	cv::Mat img_minimap;
	cv::Rect rect_minimap;
	cv::Mat img_minimap_padding;        //新增，用于匹配。部分匹配算法对边距要求严格
	cv::Rect rect_minimap_padding;
	float minimap_diameter;				//新增，小地图直径，用于做边缘剔除。由于边缘填充的存在，直径可能小于图像宽高
	cv::Point point_minimap_center;
	cv::Rect rect_avatar;
	cv::Mat img_avatar;
	cv::Rect rect_viewer;
	cv::Mat img_viewer;

	//匹配器
	std::shared_ptr<IMatcher> matcher;
	GenshinMinimapConfig config;
};

struct GenshinAvatarDirectionConfig
{
};
struct GenshinAvatarDirection
{
	float angle = 0;
	GenshinAvatarDirectionConfig config;
};

struct GenshinAvatarPositionConfig
{
	bool is_init_finish = false;
	bool is_lock_minimap_rotation = true;
	float minimap_rotation = 0;
	bool is_continuity = false;
	bool is_exist_last_match_minimap = false;
	cv::Mat img_last_match_minimap;
	bool is_use_filter = false;
	std::shared_ptr<Filter> pos_filter;
};
struct GenshinAvatarPosition
{
	cv::Point2d target_map_world_center;
	float target_map_world_scale = 1.0f;
	cv::Point2d position;
	GenshinAvatarPositionConfig config;
};

struct GenshinViewerDirectionConfig
{
	bool is_lock_minimap_rotation = true;
	float minimap_rotation = 0;
};

struct GenshinViewerDirection
{
	float angle = 0;
	GenshinViewerDirectionConfig config;
};

struct GenshinMinimapDirectionConfig
{
	bool  is_defalut = true;
	bool is_skip = true;
};

struct GenshinMinimapDirection
{
	float angle = 0;
	GenshinMinimapDirectionConfig config;
};
struct GenshinStarsConfig
{
	float check_match_star_params = 0.85f;
};
struct GenshinStars
{
	bool is_find = false;
	std::vector<cv::Point> points_star;
	GenshinStarsConfig config;
};

struct GenshinTagflagsConfig
{
	float check_match_star_params = 0.85f;
};
struct GenshinTagflags
{
	bool is_find = false;
	std::vector<cv::Point> points_star;
	GenshinTagflagsConfig config;
};

struct GenshinUIDConfig
{
	float check_match_uid_params = 0.85f;
};
struct GenshinUID
{
	int uid = 0;
	GenshinUIDConfig config;
};