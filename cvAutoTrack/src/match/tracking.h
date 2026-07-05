#pragma once
#include "utils/Utils.h"
#include <match/IMatcher.h>
#include <match/diff/DiffMatch.h>

// 特征点匹配的剔除因子，越大越宽松
constexpr double LOWE_RATIO_THRESH = 0.75;
constexpr double LOWE_RATIO_THRESH_CONTINUITY = 0.66;

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

// ============================================================
// InertialNavigator — DiffMatch 帧间相位相关惯性导航的状态管理
//
// 双空间位移追踪：
//   像素空间（小地图图像坐标）— 相位相关原生输出，永远准确
//   坐标空间（大地图坐标）    — 像素位移 × m_tracking_scale，用于位置积分
// 所有决策阈值使用像素空间，避免 m_tracking_scale 过时导致误判。
//
// 两级相位相关：
//   小步（帧间）：每帧做，保证连续性，累积误差
//   大步（关键帧）：B2c 回退时，直接对缓存帧做相位相关，积分次数少
//
// 职责：
//   1. 双空间累积位移追踪
//   2. 关键帧缓存（大步校正回退）
//   3. 传送检测（峰值突降一票否决）
//   4. 定期重试局部特征匹配以回到主模式
// ============================================================
struct InertialNavigator
{
    // ========== 帧计数 & 位移 ==========
    int consecutive_frames = 0;

    // 坐标空间（地图像素）— 用于位置积分，依赖 m_tracking_scale，可能过时
    double coord_displacement = 0.0;
    // 像素空间（小地图像素）— 相位相关原生单位，永远准确，用于决策
    double pixel_displacement = 0.0;

    // ========== 峰值 EMA ==========
    double peak_ema = 0.0;
    double prev_peak = 0.0;

    // ========== 冷却 ==========
    int last_correction_attempt = -999;

    // ========== 关键帧（B2c 大步回退） ==========
    cv::Mat keyframe;
    cv::Point2d keyframe_pos = {0, 0};
    int steps_since_keyframe = 0;
    double pixel_since_keyframe = 0.0;   // 像素空间，判断关键帧是否过旧

    // ========== 帧间小步 ==========
    cv::Mat last_frame;                  // 上一帧图像，用于帧间相位相关

    // ================================================================
    // 可调参数 — 所有距离阈值均为小地图像素空间（不依赖 scale）
    // 小地图直径约 200px，以下阈值以此为参照
    // ================================================================
    static constexpr double EMA_ALPHA = 0.3;
    static constexpr int CORRECTION_INTERVAL = 15;             // 固定间隔（帧）
    static constexpr double PIXEL_DRIFT_THRESHOLD = 80.0;      // 像素累积漂移阈值
    static constexpr double PEAK_CORRECTION_THRESHOLD = 0.35;  // EMA 低于此触发校正
    static constexpr double PEAK_VETO_THRESHOLD = 0.20;        // 一票否决最低峰值
    static constexpr double VETO_DROP_RATE = 0.50;             // 突降率 > 50% 判传送
    static constexpr int MAX_INERTIAL_FRAMES = 600;             // 预计提供最多一分钟的惯性预测
    static constexpr int CORRECTION_COOLDOWN = 5;              // 校正失败冷却

    // 大步关键帧校正参数
    static constexpr double KEYFRAME_PEAK_THRESHOLD = 0.20;    // 大步相位相关最低峰值

    // ========== 方法 ==========

    /**
     * @brief 记录一次小步惯性导航结果
     * @param peak         帧间相位相关峰值 [0,1]
     * @param coord_dist   本帧坐标空间位移（地图像素）
     * @param pixel_dist   本帧像素空间位移（小地图像素）
     */
    void record(double peak, double coord_dist, double pixel_dist)
    {
        consecutive_frames++;
        coord_displacement += coord_dist;
        pixel_displacement += pixel_dist;
        steps_since_keyframe++;
        pixel_since_keyframe += pixel_dist;
        if (consecutive_frames == 1)
            peak_ema = peak;
        else
            peak_ema = EMA_ALPHA * peak + (1.0 - EMA_ALPHA) * peak_ema;
        prev_peak = peak;
    }

    bool hasKeyframe() const { return !keyframe.empty(); }

    void captureKeyframe(const cv::Mat& frame, cv::Point2d pos)
    {
        keyframe = frame.clone();
        keyframe_pos = pos;
        steps_since_keyframe = 0;
        pixel_since_keyframe = 0.0;
    }

    void releaseKeyframe() { keyframe = cv::Mat(); }

    bool hasLastFrame() const { return !last_frame.empty(); }
    void updateLastFrame(const cv::Mat& frame) { last_frame = frame.clone(); }
    void releaseLastFrame() { last_frame = cv::Mat(); }

    /// 是否需要重试校正？（带冷却，硬上限不受冷却限制）
    bool needsCorrection() const
    {
        // 纯惯性太久，无论当前状态如何均强制校正
        if (consecutive_frames >= MAX_INERTIAL_FRAMES)
            return true;
        if (consecutive_frames - last_correction_attempt < CORRECTION_COOLDOWN)
            return false;
        return consecutive_frames >= CORRECTION_INTERVAL
            || pixel_displacement >= PIXEL_DRIFT_THRESHOLD
            || peak_ema < PEAK_CORRECTION_THRESHOLD;
    }

    void markCorrectionAttempted()
    {
        last_correction_attempt = consecutive_frames;
    }

    /// 一票否决
    bool needsVeto(double current_peak) const
    {
        if (consecutive_frames < 3) return false;
        if (current_peak >= PEAK_VETO_THRESHOLD) return false;
        double drop_rate = (prev_peak - current_peak) /
                           (std::max)(prev_peak, 1e-6);
        return drop_rate > VETO_DROP_RATE;
    }

    void reset()
    {
        consecutive_frames = 0;
        coord_displacement = 0.0;
        pixel_displacement = 0.0;
        peak_ema = 0.0;
        prev_peak = 0.0;
        last_correction_attempt = -999;
        releaseKeyframe();
        releaseLastFrame();
    }
};

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
    InertialNavigator m_inertial;       // 惯性导航状态（校正决策 + 传送检测）

	bool m_isInit = false;
	bool m_isContinuity = false;

	bool m_is_success_match = false;

	void setMap(cv::Mat gi_map);
	/**
	 * @brief 设置小地图图像
	 * @param minimap GenshinMinimap 结构体
	 *        - img_minimap_padding 用于特征点匹配
	 *        - img_minimap 用于相位相关
	 *        - minimap_diameter 用于边缘剔除
	 */
	void setMiniMap(const GenshinMinimap& minimap);

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
