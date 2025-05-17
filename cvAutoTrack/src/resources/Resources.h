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
    const cv::Point2d map_relative_center = { 6668, 3662 }; // 天理坐标中点
    // 地图中图片像素与天理坐标系的比例
    const double map_relative_scale = 3.413333; // 天理坐标缩放
    // 手柄模式相对于键鼠模式ui大小的缩放值的倒数
    const double controller_ui_scale = 1.2;
public:
    void install();
    void release();
public:
    //void get_map_keypoint_cache();
    bool map_is_embedded();
private:
    bool is_installed = false;
};

class MapKeypointCache {
public:
    std::string bulid_time;
    std::string bulid_version;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    std::string bulid_version_end;

    MapKeypointCache() {}

    MapKeypointCache(std::string bulid_time,
        std::string bulid_version,
        std::vector<cv::KeyPoint> keyPoints,
        cv::Mat descriptors) :
        bulid_time(bulid_time), bulid_version(bulid_version),
        keyPoints(keyPoints), descriptors(descriptors), bulid_version_end(bulid_version) {}

    void serialize(std::string outfileName);
    /**
     * @brief 解码缓存
     * @param infileName 输入文件名
     * @param version_only 是否只解析版本，防止数据结构不一致而崩溃
     */
    void deSerialize(std::string infileName, bool version_only = false);
};

bool save_map_keypoint_cache(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
bool load_map_keypoint_cache(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

bool get_map_keypoint(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);