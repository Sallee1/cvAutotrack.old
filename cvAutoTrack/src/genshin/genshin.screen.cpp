#include "pch.h"
#include "genshin.screen.h"
#include "genshin.minimap.h"
#include <chrono>

namespace TianLi::Genshin {
    void init_screen_frames(GenshinScreen& out_genshin_screen);

    /**
     * @brief 从 HDR 帧左上角 ROI 采样白点值（只扫 R 通道）
     */
    inline double detect_white_point(const cv::Mat& hdr_rgba)
    {
        CV_Assert(hdr_rgba.channels() == 4 && hdr_rgba.depth() == CV_32F);
        const int W = hdr_rgba.cols;
        const int H = hdr_rgba.rows;
        const cv::Rect tl_roi(0, 0, static_cast<int>(W * 0.22), static_cast<int>(H * 0.22));

        cv::Mat roi_view(hdr_rgba, tl_roi);
        cv::Mat r_chan(roi_view.size(), CV_32FC1);
        int from_to[] = { 0, 0 };
        cv::mixChannels(&roi_view, 1, &r_chan, 1, from_to, 1);
        double max_val;
        cv::minMaxLoc(r_chan, nullptr, &max_val);
        return (std::min)(max_val, 6.0);
    }

    /**
     * @brief HDR→SDR 色调映射（单 ROI）
     * @param hdr_rgba  输入 RGBA 32F/16F 图像（ROI 视图）
     * @param white_point 白点值
     * @return          BGRA 8UC4 SDR 图像
     */
    cv::Mat tone_map_hdr_to_sdr(const cv::Mat& hdr_rgba, double white_point)
    {
        CV_Assert(hdr_rgba.channels() == 4);

        cv::Mat hdr_float;
        if (hdr_rgba.depth() == CV_16F || hdr_rgba.depth() != CV_32F)
            hdr_rgba.convertTo(hdr_float, CV_32F);
        else
            hdr_float = hdr_rgba;

        const float inv_max = 1.0f / static_cast<float>(white_point);
        const float gamma   = 1.0f / 2.2f;

        hdr_float.forEach<cv::Vec4f>([inv_max, gamma](cv::Vec4f& p, const int*) {
            p[0] = std::pow(p[0] * inv_max, gamma);   // B
            p[1] = std::pow(p[1] * inv_max, gamma);   // G
            p[2] = std::pow(p[2] * inv_max, gamma);   // R
            p[3] = 1.0f;
        });

        cv::Mat sdr_bgra;
        hdr_float.convertTo(sdr_bgra, CV_8UC4, 255.0);
        cv::cvtColor(sdr_bgra, sdr_bgra, cv::COLOR_RGBA2BGRA);
        return sdr_bgra;
    }

    /**
     * @brief 预处理原始帧（仅处理非 HDR 路径：alpha 剥离 + 亮度提升）
     */
    void preprocess_raw_frame(cv::Mat& mat);

    bool get_genshin_screen(const GenshinHandle& genshin_handle, GenshinScreen& out_genshin_screen,
                            GenshinMinimap* out_minimap)
    {
        static HBITMAP hBmp;

        auto& giRect = genshin_handle.rect;
        auto& giRectClient = genshin_handle.rect_client;
        auto& giFrame = out_genshin_screen.img_screen;

        auto now_time = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now_time - out_genshin_screen.last_time).count() > 20 || giFrame.empty())
        {
            out_genshin_screen.last_time = now_time;
            cv::Mat raw_frame;
            if (!genshin_handle.config.frame_source->get_frame(raw_frame))
            {
                return false;
            }

            if (raw_frame.depth() == CV_16F)
            {
                raw_frame.convertTo(raw_frame, CV_32F);
            }
            cv::resize(raw_frame, giFrame, genshin_handle.size_frame);
        }

        {
            if (giFrame.empty()) return false;

            out_genshin_screen.rect_client = cv::Rect(giRect.left, giRect.top, giRectClient.right - giRectClient.left, giRectClient.bottom - giRectClient.top);

            if (giFrame.depth() == CV_32F)
            {
                // HDR 路径：小地图首次成功检测后白点亮度即锁定，不再重复计算
                if (!out_genshin_screen.hdr_cache.white_point_locked)
                {
                    out_genshin_screen.hdr_cache.white_point = detect_white_point(giFrame);
                }
            }
            else
            {
                // SDR 路径：alpha 剥离 + 亮度提升
                preprocess_raw_frame(giFrame);
            }
            init_screen_frames(out_genshin_screen);

            // ---- 截图后立即执行小地图定位（避免后续单独调用）----
            if (out_minimap)
            {
                out_genshin_screen.config.is_used_alpha =
                    (genshin_handle.config.frame_source->type != tianli::frame::frame_source::source_type::window_graphics &&
                     !genshin_handle.config.is_force_used_no_alpha);

                if (find_minimap(out_genshin_screen, *out_minimap))
                {                    
                    out_minimap->is_minimap_fresh = true;
                    out_genshin_screen.hdr_cache.white_point_locked = true;                    
                    if (out_genshin_screen.config.is_controller_mode)
                    {
                        const auto& s = out_genshin_screen.config.controller_ui_scale;
                        cv::resize(out_minimap->img_minimap, out_minimap->img_minimap, cv::Size(),
                            1.0 / s, 1.0 / s, cv::INTER_AREA);
                    }
                }
            }
        }
        return true;
    }

    void init_screen_frames(GenshinScreen& out_genshin_screen)
    {
        auto& giFrame = out_genshin_screen.img_screen;
        const bool is_hdr = (giFrame.depth() == CV_32F);
        const double white_point = is_hdr ? out_genshin_screen.hdr_cache.white_point : 0.0;

        // 根据宽高比裁剪帧（原神客户端左上角对齐）
        int x_size = giFrame.cols;
        int y_size = giFrame.rows;

        float screen_ratio = static_cast<float>(x_size) / static_cast<float>(y_size);
        if (screen_ratio > (16.0f / 9.0f))
        {
            x_size = static_cast<int>(y_size * (16.0f / 9.0f));
            y_size = y_size;
        }
        else if (screen_ratio < (16.0f / 9.0f))
        {
            x_size = x_size;
            y_size = static_cast<int>(x_size * (9.0f / 16.0f));
        }

        // 小地图标定可能性区域（在 tl_sdr 坐标系内，原点 (0,0)）
        out_genshin_screen.rects.icon_sight_maybe = cv::Rect(
            static_cast<int>(x_size * 0.10), 0,
            static_cast<int>(x_size * 0.12), static_cast<int>(y_size * 0.10));

        // UID 区域（giFrame 坐标系）
        int UID_Rect_x = cvCeil(x_size - x_size * (1.0 - 0.865));
        int UID_Rect_y = cvCeil(y_size - 1080.0 * (1.0 - 0.9755));
        int UID_Rect_w = cvCeil(1920 * 0.11);
        int UID_Rect_h = cvCeil(1920 * 0.0938 * 0.11);
        out_genshin_screen.rects.uid = cv::Rect(UID_Rect_x, UID_Rect_y, UID_Rect_w, UID_Rect_h);
        out_genshin_screen.rects.uid += cv::Size2i{ (giFrame.cols - x_size), (giFrame.rows - y_size) };

        // ---- 一次 tone map，两块 8UC4 clone（后续 UI 检测全从 SDR 读，零 HDR 判断）----
        const cv::Rect tl_roi(0, 0, x_size / 4, y_size / 4);
        if (is_hdr)
        {
            out_genshin_screen.imgs.minimap_maybe = tone_map_hdr_to_sdr(giFrame(tl_roi).clone(), white_point);
            out_genshin_screen.imgs.uid_maybe = tone_map_hdr_to_sdr(giFrame(out_genshin_screen.rects.uid).clone(), white_point);
        }
        else
        {
            out_genshin_screen.imgs.minimap_maybe = giFrame(tl_roi).clone();
            out_genshin_screen.imgs.uid_maybe = giFrame(out_genshin_screen.rects.uid).clone();
        }
    }

    void preprocess_raw_frame(cv::Mat& mat)
    {
        // 仅处理非 HDR 路径（8U 输入）
        bool is_grayscale = (mat.channels() == 1);
        bool has_alpha = (mat.channels() == 4);

        cv::Mat bgr_mat;
        cv::Mat alpha_mat;
        std::vector<cv::Mat> bgr_a_channels;
        if (has_alpha)
        {
            bgr_mat = cv::Mat(mat.size(), CV_8UC3);
            alpha_mat = cv::Mat(mat.size(), CV_8UC1);
            bgr_a_channels = std::vector{ bgr_mat, alpha_mat };
            cv::mixChannels(mat, bgr_a_channels, { 0,0,1,1,2,2,3,3 });
        }
        else {
            bgr_mat = mat;
        }

        double min_col, max_col;
        cv::Mat grayscale_mat;
        if (!is_grayscale) {
            cv::cvtColor(bgr_mat, grayscale_mat, cv::COLOR_RGB2GRAY);
        }

        cv::minMaxLoc(grayscale_mat, &min_col, &max_col);
        if (max_col > 250)
        {
            return;
        }
        cv::Mat f_mat;
        bgr_mat.convertTo(f_mat, CV_32FC3);
        f_mat = f_mat * (255.0 / max_col);
        f_mat.convertTo(bgr_mat, CV_8UC3);

        if (has_alpha)
        {
            cv::mixChannels(bgr_a_channels, mat, { 0,0,1,1,2,2,3,3 });
        }
        else {
            mat = bgr_mat;
        }
    }
}
