#include "pch.h"
#include "genshin.screen.h"
#include <chrono>

namespace TianLi::Genshin {
    void init_screen_frames(GenshinScreen& out_genshin_screen);


    inline cv::Mat tone_map_hdr_to_sdr_bgra(const cv::Mat& hdr_rgba)
    {
        cv::Mat hdr_float;
        hdr_rgba.convertTo(hdr_float, CV_32FC4);
        cv::cvtColor(hdr_float, hdr_float, cv::COLOR_RGBA2BGRA);
        std::vector<cv::Mat> channels;
        cv::split(hdr_float, channels);

        //采集白点值，先尝试线性压缩
        //原神是假HDR，UI的最亮点可以作为白点参考
        for (int i = 0; i < 3; ++i)
        {
            cv::max(channels[i], 0.0f, channels[i]);
            channels[i] = channels[i] / (1.0f + channels[i]); // Reinhard tone mapping
            cv::pow(channels[i], 1.0 / 2.2, channels[i]);     // gamma to SDR
        }
        channels[3] = cv::Mat::ones(channels[3].size(), channels[3].type());

        cv::Mat merged;
        cv::merge(channels, merged);
        cv::Mat sdr_bgra;
        merged.convertTo(sdr_bgra, CV_8UC4, 255.0);
        return sdr_bgra;
    }

    /**
     * @brief 预处理原始帧
     * @param mat 待处理的帧
     */
    void preprocess_raw_frame(cv::Mat& mat);

    bool get_genshin_screen(const GenshinHandle& genshin_handle, GenshinScreen& out_genshin_screen)
    {
        static HBITMAP hBmp;

        //auto& giHandle = genshin_handle.handle;
        auto& giRect = genshin_handle.rect;
        auto& giRectClient = genshin_handle.rect_client;
        //auto& giScale = genshin_handle.scale;
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
            cv::resize(raw_frame, giFrame, genshin_handle.size_frame);
        }

        {
            if (giFrame.empty())return false;

            out_genshin_screen.rect_client = cv::Rect(giRect.left, giRect.top, giRectClient.right - giRectClient.left, giRectClient.bottom - giRectClient.top);
            preprocess_raw_frame(giFrame);
            init_screen_frames(out_genshin_screen);
        }
        return true;
    }

    void init_screen_frames(GenshinScreen& out_genshin_screen)
    {
        auto& giFrame = out_genshin_screen.img_screen;

        // 根据宽高比裁剪帧
        // 原神客户端左上角对齐，当前屏幕宽高比大于16:9时，高度不变填充宽度
        // 宽高比小于16:9时，宽度不变填充高度
        // 为了正确获取左上角的小地图的可能性区域
        // 需要左上角对齐裁剪填充的部分
        int x_size = giFrame.cols;
        int y_size = giFrame.rows;

        float screen_ratio = static_cast<float>(x_size) / static_cast<float>(y_size);
        if(screen_ratio > (16.0f / 9.0f))
        {
            // 宽屏，裁剪左右
            x_size = static_cast<int>(y_size * (16.0f / 9.0f));
            y_size = y_size;
        }
        else if(screen_ratio < (16.0f / 9.0f))
        {
            // 高屏，裁剪上下
            x_size = x_size;
            y_size = static_cast<int>(x_size * (9.0f / 16.0f));
        }


        // 小地图标定可能性区域计算参数
        int icon_sight_mayArea_left = static_cast<int>(x_size * 0.10);
        int icon_sight_mayArea_top = 0;
        int icon_sight_mayArea_width = static_cast<int>(x_size * 0.12);
        int icon_sight_mayArea_height = static_cast<int>(y_size * 0.10);
        // 小地图标定可能性区域
        cv::Rect Area_icon_sight_mayArea(
            icon_sight_mayArea_left,
            icon_sight_mayArea_top,
            icon_sight_mayArea_width,
            icon_sight_mayArea_height);
        out_genshin_screen.rects.icon_sight_maybe = Area_icon_sight_mayArea;

        // 小地图可能性区域计算参数
        int miniMap_mayArea_left = 0;
        int miniMap_mayArea_top = 0;
        int miniMap_mayArea_width = static_cast<int>(x_size * 0.18);
        int miniMap_mayArea_height = static_cast<int>(y_size * 0.22);
        // 小地图可能性区域
        cv::Rect Area_MiniMap_mayArea(
            miniMap_mayArea_left,
            miniMap_mayArea_top,
            miniMap_mayArea_width,
            miniMap_mayArea_height);
        out_genshin_screen.rects.minimap_maybe = Area_MiniMap_mayArea;

        // UID可能性区域计算参数
        // UID在右下角，所以需要包括填充区域的偏移
        int UID_mayArea_left = static_cast<int>(x_size * 0.88) + (x_size - x_size);
        int UID_mayArea_top = static_cast<int>(y_size * 0.97) + (y_size - y_size);
        int UID_mayArea_width = x_size - UID_mayArea_left + (x_size - x_size);
        int UID_mayArea_height = y_size - UID_mayArea_top + (y_size - y_size);
        // UID可能性区域
        cv::Rect Area_UID_mayArea(
            UID_mayArea_left,
            UID_mayArea_top,
            UID_mayArea_width,
            UID_mayArea_height);
        out_genshin_screen.rects.uid_maybe = Area_UID_mayArea;
        out_genshin_screen.rects.uid_maybe += cv::Size2i{ (giFrame.cols - x_size),(giFrame.rows - y_size) };

        int UID_Rect_x = cvCeil(x_size - x_size * (1.0 - 0.865));
        int UID_Rect_y = cvCeil(y_size - 1080.0 * (1.0 - 0.9755));
        int UID_Rect_w = cvCeil(1920 * 0.11);
        int UID_Rect_h = cvCeil(1920 * 0.0938 * 0.11);
        out_genshin_screen.rects.uid = cv::Rect(UID_Rect_x, UID_Rect_y, UID_Rect_w, UID_Rect_h);
        out_genshin_screen.rects.uid += cv::Size2i{ (giFrame.cols - x_size),(giFrame.rows - y_size) };

        // 获取maybe区域
        out_genshin_screen.imgs.icon_sight_maybe = giFrame(out_genshin_screen.rects.icon_sight_maybe);
        out_genshin_screen.imgs.minimap_maybe = giFrame(out_genshin_screen.rects.minimap_maybe);
        out_genshin_screen.imgs.uid_maybe = giFrame(out_genshin_screen.rects.uid_maybe);
        out_genshin_screen.imgs.uid = giFrame(out_genshin_screen.rects.uid);
    }

    void preprocess_raw_frame(cv::Mat& mat)
    {
        //提取alpha通道
        bool is_grayscale = (mat.channels() == 1);
        bool has_alpha = (mat.channels() == 4);

        cv::Mat bgr_mat;
        cv::Mat alpha_mat;
        std::vector<cv::Mat> bgr_a_channels;
        if (has_alpha)
        {
            bgr_mat = cv::Mat(mat.size(), CV_8UC3);
            alpha_mat = cv::Mat(mat.size(), CV_8UC1);
            bgr_a_channels = std::vector{ bgr_mat,alpha_mat };
            cv::mixChannels(mat, bgr_a_channels, { 0,0,1,1,2,2,3,3 });
        }
        else {
            bgr_mat = mat;
        }

        //目前主要工作是提亮，因为画面中有标准白，不是问题，但没有标准黑是个问题
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

        return;
    }
}
