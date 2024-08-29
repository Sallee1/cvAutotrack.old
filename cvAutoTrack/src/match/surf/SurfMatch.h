#pragma once
#include "utils/Utils.h"

// 特征点匹配的剔除因子，越大越严格
constexpr double SURF_MATCH_RATIO_THRESH = 0.66;
// 地图和小地图野外的缩放比例，（大地图 / 小地图野外）得到，注意城镇内小地图是野外的两倍，所以是城镇内比例是1.3/2
constexpr double MAP_BOTH_SCALE_RATE = 1.0;
// 地图中取小部分区域的半径，目前为小地图标准半径
constexpr int DEFAULT_SOME_MAP_SIZE_R = 150;

class Match
{
public:
    struct KeyMatPoint
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        bool empty() { return keypoints.size() == 0; }
        auto size() { return keypoints.size(); }
    };
public:
    Match(double hessian_threshold = 1, int octaves = 1, int octave_layers = 1, bool extended = false, bool upright = true);
    ~Match() = default;
public:
    cv::Ptr<cv::xfeatures2d::SURF> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    KeyMatPoint query;
    KeyMatPoint train;
public:
    std::vector<std::vector<cv::DMatch>> match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors);
    std::vector<std::vector<cv::DMatch>> match(KeyMatPoint& query_key_mat_point, KeyMatPoint& train_key_mat_point);
    bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    bool detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point);
};

class SurfMatch
{
    cv::Mat _mapMat;
    cv::Mat _miniMapMat;
    cv::Mat _miniMapLastMat;

    cv::Point2d pos;
    cv::Point2d last_pos;		// 上一次匹配的地点，匹配失败，返回上一次的结果
    cv::Rect rect_continuity_map;
public:
    SurfMatch() = default;
    ~SurfMatch() = default;

public:
    Match matcher;

    Match::KeyMatPoint map, some_map, mini_map;

    bool isInit = false;
    bool isContinuity = false;

    int continuity_retry = 0;		//局部匹配重试次数
    const int max_continuity_retry = 3;		//最大重试次数

    bool is_success_match = false;

    void setMap(cv::Mat gi_map);
    void setMiniMap(cv::Mat miniMapMat);

    void Init();
    void Init(std::vector<cv::KeyPoint>& gi_map_keypoints, cv::Mat& gi_map_descriptors);
    void UnInit();
    void match();

    cv::Point2d match_continuity(bool& calc_continuity_is_faile);

    cv::Point2d match_no_continuity(bool& calc_is_faile);

    //全图匹配
    //cv::Point2d match_all_map(bool& calc_is_faile,double& stdev, double minimap_scale_param = 1.0);

    cv::Point2d getLocalPos();
    bool getIsContinuity();

private:
    bool isMatchAllMap = true;
};
