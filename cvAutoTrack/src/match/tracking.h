#pragma once
#include "utils/Utils.h"
#include <match/IMatcher.h>

// 特征点匹配的剔除因子，越大越宽松
constexpr double LOWE_RATIO_THRESH = 0.7;
constexpr double LOWE_RATIO_THRESH_CONTINUITY = 0.8;

// 地图和小地图野外的缩放比例，（大地图 / 小地图野外）得到，注意城镇内小地图是野外的两倍，所以是城镇内比例是1.3/2
constexpr double MAP_BOTH_SCALE_RATE = 1.0;
// 地图中取小部分区域的半径，目前为小地图标准半径
constexpr int DEFAULT_SOME_MAP_SIZE_R = 300;

class Tracking
{
    cv::Mat _mapMat;
    cv::Mat _miniMapMat;
    cv::Mat _miniMapLastMat;

    cv::Point2d pos;
    cv::Point2d last_pos;		// 上一次匹配的地点，匹配失败，返回上一次的结果
    cv::Rect rect_continuity_map;
public:
    Tracking() = default;
    ~Tracking() = default;

public:
    std::shared_ptr<IMatcher> m_matcher = nullptr;

    IMatcher::KeyMatPoint map, some_map, mini_map;

    bool isInit = false;
    bool isContinuity = false;

    int continuity_retry = 0;		//局部匹配重试次数
    const int max_continuity_retry = 3;		//最大重试次数

    bool is_success_match = false;

    void setMap(cv::Mat gi_map);
    void setMiniMap(cv::Mat miniMapMat);

    bool Init(const std::shared_ptr<IMatcher>& matcher);
    bool Init(const std::shared_ptr<IMatcher>& matcher, std::vector<cv::KeyPoint>&& gi_map_keypoints, cv::Mat&& gi_map_descriptors);
    void UnInit();
    void match();

    cv::Point2d getLocalPos();
    bool getIsContinuity();

private:

    cv::Point2d match_continuity(bool& calc_continuity_is_faile);

    cv::Point2d match_no_continuity(bool& calc_is_faile);

    cv::Point2d match_impl(const cv::Mat& img_scene, const IMatcher::KeyMatPoint& keypoint_scene, const cv::Mat& img_object, const IMatcher::KeyMatPoint& keypoint_object, bool& calc_is_faile);

    cv::Point2d cleanAndComputePos_Old(std::vector<cv::Point2f>& good_matched_scene, std::vector<cv::Point2f>& good_matched_object, bool& calc_is_faile);

    //全图匹配
    //cv::Point2d match_all_map(bool& calc_is_faile,double& stdev, double minimap_scale_param = 1.0);
    bool isMatchAllMap = true;
};
