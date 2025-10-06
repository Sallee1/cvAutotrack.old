#pragma once
#include "utils/Utils.h"
#include <match/IMatcher.h>

// 特征点匹配的剔除因子，越大越宽松
constexpr double LOWE_RATIO_THRESH = 0.75;
constexpr double LOWE_RATIO_THRESH_CONTINUITY = 0.66;

// 地图和小地图野外的缩放比例，（大地图 / 小地图野外）得到，注意城镇内小地图是野外的两倍，所以是城镇内比例是1.3/2
constexpr double MAP_BOTH_SCALE_RATE = 1.0;
// 地图中取小部分区域的半径，目前为小地图标准半径
constexpr int DEFAULT_SOME_MAP_SIZE_R = 200;
// 小地图边缘过滤比例
constexpr float MINIMAP_BORDER_CROP_RATIO = 0.95f;

struct MapKeypointCache;
struct KeypointGridLSH;

class Tracking
{
	cv::Mat _mapMat;
	cv::Mat _miniMapMat;
	cv::Mat _miniMapLastMat;
	float _miniMapDiameter = 0;

	cv::Point2d pos;
	cv::Point2d last_pos;		// 上一次匹配的地点，匹配失败，返回上一次的结果
	cv::Rect rect_continuity_map;
public:
	Tracking() = default;
	~Tracking() = default;

public:
	std::shared_ptr<IMatcher> m_matcher = nullptr;

	IMatcher::KeyMatPoint map_kp, some_map_kp, mini_map_kp;
	std::unique_ptr<KeypointGridLSH> m_lsh_index;

	bool isInit = false;
	bool isContinuity = false;

	int continuity_retry = 0;		//局部匹配重试次数
	const int max_continuity_retry = 3;		//最大重试次数

	bool is_success_match = false;

	void setMap(cv::Mat gi_map);
	/**
	 * @brief 设置小地图图像
	 * @param miniMapMat 小地图图像
	 * @param diameter 小地图直径，为0表示将图像大小视为直径，用于进行边界裁剪
	 */
	void setMiniMap(cv::Mat miniMapMat, float diameter = 0);

	bool Init(const std::shared_ptr<IMatcher>& matcher);
	bool Init(const std::shared_ptr<IMatcher>& matcher, std::vector<cv::KeyPoint>&& gi_map_keypoints, cv::Mat&& gi_map_descriptors);
	bool Init(const std::shared_ptr<IMatcher>& matcher, MapKeypointCache&& map_keypoints_cache);
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
