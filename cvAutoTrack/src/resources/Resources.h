#pragma once
#include <match/type/MatchType.h>

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
	cv::Mat PaimonTemplate;
	cv::Mat IconSightTemplate;
	cv::Mat IconQuestTemplate;
	cv::Mat StarTemplate;
	cv::Mat MapTemplate;
	cv::Mat UID;
	cv::Mat UIDnumber[10];

	// 天理坐标映射关系参数 地图中心
	// 地图中天理坐标中心的像素坐标
	const cv::Point2f map_relative_center = { 6668, 3662 }; // 天理坐标中点
	// 地图中图片像素与天理坐标系的比例
	const float map_relative_scale = 3.413333f; // 天理坐标缩放
	// 手柄模式相对于键鼠模式ui大小的缩放值的倒数
	const float controller_ui_scale = 1.2f;
	//lsh的块大小
	int lsh_cell_size = 600;
public:
	void install();
	void release();
public:
	bool map_is_embedded();
private:
	bool is_installed = false;
};