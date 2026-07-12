#pragma once
#include "utils/Utils.h"
#include <match/IMatcher.h>
#include <match/diff/DiffMatch.h>

// 特征点匹配的剔除因子，越大越宽松
constexpr double LOWE_RATIO_THRESH = 0.75;
constexpr double LOWE_RATIO_THRESH_CONTINUITY = 0.75;

// 地图和小地图野外的缩放比例，（大地图 / 小地图野外）得到，注意城镇内小地图是野外的两倍，所以是城镇内比例是1.3/2
constexpr double MAP_BOTH_SCALE_RATE = 1.0;
// 地图中取小部分区域的半径，目前为小地图标准半径
constexpr int DEFAULT_SOME_MAP_SIZE_R = 200;
// 小地图边缘过滤比例
constexpr float MINIMAP_BORDER_CROP_RATIO = 0.95f;
// 小地图标准尺寸
constexpr double MINIMAP_DIAMETER = 200;

struct MapKeypointCache;
struct KeypointGridLSH;
struct GenshinMinimap;

#include <match/InertialNavigator.h>

class Tracking
{
	cv::Mat m_mapMat;
	cv::Mat m_miniMapMat;		    // 带遮罩+padding 版，用于特征点匹配
	cv::Mat m_miniMapCenter;		// 原始裁剪版（无遮罩无padding），用于相位相关
	float m_miniMapDiameter = 0;

	cv::Point2d m_pos;
	cv::Point2d m_last_pos;		// 上一次匹配的地点，匹配失败，返回上一次的结果
public:
	Tracking() = default;
	~Tracking() = default;

public:
	std::shared_ptr<IMatcher> m_matcher = nullptr;

	IMatcher::KeyMatPoint m_map_kp;
	std::unique_ptr<KeypointGridLSH> m_lsh_index;

    double m_tracking_scale = 1.0;      // 缩放参考，用于惯性导航（由特征匹配更新）
    double m_tracking_scale_smooth = 1.0; // 平滑后的缩放系数
    int m_scale_sample_count = 0;        // 平滑窗口样本数

    void updateTrackingScale(double raw_scale)
    {
        static constexpr double JUMP_THRESHOLD = 0.10;   // 相对于平滑值变化超 ±10%，判为缩放切换
        static constexpr int MAX_WINDOW = 10;              // 最大平滑窗口

        // 首次直接接受
        if (m_scale_sample_count == 0)
        {
            m_tracking_scale_smooth = raw_scale;
            m_scale_sample_count = 1;
            m_tracking_scale = raw_scale;
            return;
        }

        // 差异超 10% → 缩放已切换，重新累积
        double diff_ratio = std::abs(raw_scale - m_tracking_scale_smooth) / m_tracking_scale_smooth;
        if (diff_ratio > JUMP_THRESHOLD)
        {
            m_tracking_scale_smooth = raw_scale;
            m_scale_sample_count = 1;
            m_tracking_scale = raw_scale;
            return;
        }

        // 正常平滑累积（运行平均）
        int n = std::min(m_scale_sample_count, MAX_WINDOW);
        m_tracking_scale_smooth = (m_tracking_scale_smooth * n + raw_scale) / (n + 1);
        m_scale_sample_count++;
        m_tracking_scale = m_tracking_scale_smooth;
    }

    InertialNavigator m_inertial;       // 惯性导航状态（校正决策 + 传送检测）

	bool m_isInit = false;
	bool m_isContinuity = false;

	bool m_is_success_match = false;

#ifdef _CVAT_PURE_INS
	// ========== 调试：惯性导航与特征匹配并行对比 ==========
	std::ofstream m_debug_csv;          // CSV 输出文件
	int m_debug_step = 0;               // 惯性步数计数器
	cv::Point2d m_debug_local_pos = { NAN, NAN }; // 局部匹配结果（用于 CSV）
	bool m_debug_local_ok = false;      // 局部匹配是否成功
#endif

	void setMap(cv::Mat gi_map);
	/**
	 * @brief 设置小地图图像
	 * @param minimap GenshinMinimap 结构体
	 *        - img_minimap_padding 用于特征点匹配
	 *        - img_minimap 用于相位相关
	 *        - minimap_diameter 用于边缘剔除
	 */
	void setMiniMap(const GenshinMinimap& minimap);

    void setMatchAllMapNext();

	bool Init(const std::shared_ptr<IMatcher>& matcher);
	bool Init(const std::shared_ptr<IMatcher>& matcher, int cols, int rows, std::vector<cv::KeyPoint>&& gi_map_keypoints, cv::Mat&& gi_map_descriptors);
	bool Init(const std::shared_ptr<IMatcher>& matcher, MapKeypointCache&& map_keypoints_cache);
	void UnInit();
	void match();

	cv::Point2d getLocalPos();
	bool getIsContinuity();

private:

	cv::Point2d match_continuity(bool& calc_continuity_is_faile);

	cv::Point2d match_no_continuity(bool& calc_is_faile);

	cv::Point2d match_impl(const cv::Mat& img_scene, const IMatcher::KeyMatPoint& keypoint_scene, const cv::Mat& img_object, const IMatcher::KeyMatPoint& keypoint_object, bool& calc_is_faile);

	cv::Point2d cleanAndComputePos_Old(std::vector<cv::Point2f>& good_matched_scene,bool& calc_is_faile);

	//全图匹配
	//cv::Point2d match_all_map(bool& calc_is_faile,double& stdev, double minimap_scale_param = 1.0);
	bool m_isMatchAllMap = true;
};
