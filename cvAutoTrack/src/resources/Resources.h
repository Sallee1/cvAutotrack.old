#pragma once
#include <filesystem>
#include <match/type/MatchType.h>

namespace fs = std::filesystem;

//图片资源 加载类
class Resources
{
private:
	Resources();
public:
	~Resources();

	Resources(const Resources&) = delete;
	Resources& operator=(const Resources&) = delete;
	static Resources& getInstance();

public:
	std::map<std::pair<int, int>, cv::Mat> MapBlockCache;

public:
	cv::Mat IconSightTemplate;
	cv::Mat IconQuestTemplate;
	cv::Mat UID;
	cv::Mat UIDnumber[10];
    // 缓存路径

    struct {
#ifdef _CVAT_DEBUG
        const fs::path cvAutoTrack_Cache = "cvAutoTrack_Cache.debug.xml";
        const fs::path cvAutoTrack_Cache_flann = "cvAutoTrack_Cache.debug.flann";
        const fs::path cvAutoTrack_Cache_faiss = "cvAutoTrack_Cache.debug.faiss";
#else
        const fs::path cvAutoTrack_Cache = "cvAutoTrack_Cache.xml";
        const fs::path cvAutoTrack_Cache_flann = "cvAutoTrack_Cache.flann";
        const fs::path cvAutoTrack_Cache_faiss = "cvAutoTrack_Cache.faiss";
#endif
    } CachePath;

	// 手柄模式相对于键鼠模式ui大小的缩放值的倒数
	const float controller_ui_scale = 1.2f;
	//lsh的块大小
	int lsh_cell_size = 600;

    //调试参数
	cv::Mat DebugMapTemplate;
    struct {
        cv::Point2d offset = {0,0};
    } DebugParams;
public:
	void install();
	void release();
public:
	bool map_is_embedded();

    static fs::path getDllPath();
private:
	bool is_installed = false;
};