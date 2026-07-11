#pragma once
#include <opencv2/opencv.hpp>

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
    static constexpr double PIXEL_DRIFT_THRESHOLD = 80.0;      // 像素累积漂移阈值
    static constexpr double PEAK_VETO_THRESHOLD = 0.60;        // 相位匹配门限
    static constexpr int MAX_INERTIAL_FRAMES = 600;             // 惯性导航最大持续帧数，超限切全局匹配
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
        pixel_displacement = 0.0;
        coord_displacement = 0.0;
    }

    void releaseKeyframe() { keyframe = cv::Mat(); }

    bool hasLastFrame() const { return !last_frame.empty(); }
    void updateLastFrame(const cv::Mat& frame) { last_frame = frame.clone(); }
    void releaseLastFrame() { last_frame = cv::Mat(); }

    /// 是否需要大步校正？（仅漂移超阈值时触发，避免退化为逐步更新）
    bool needsCorrection() const
    {
        if (consecutive_frames - last_correction_attempt < CORRECTION_COOLDOWN)
            return false;
        return pixel_displacement >= PIXEL_DRIFT_THRESHOLD;
    }

    void markCorrectionAttempted()
    {
        last_correction_attempt = consecutive_frames;
    }

    /// 一票否决
    /// peak 低于 PEAK_VETO_THRESHOLD 即直接否决（无论是否突降），
    /// 避免持续低峰值时卡在惯性模式不切全局匹配。
    bool needsVeto(double current_peak) const
    {
        return current_peak < PEAK_VETO_THRESHOLD;
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
