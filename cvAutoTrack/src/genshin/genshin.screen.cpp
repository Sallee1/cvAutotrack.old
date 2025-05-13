#include "pch.h"
#include "genshin.screen.h"
#include <chrono>

namespace TianLi::Genshin {
    void init_screen_frames(GenshinScreen& out_genshin_screen);

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
            if (!genshin_handle.config.frame_source->get_frame(giFrame))
            {
                return false;
            }
        }

        {
            if (giFrame.empty())return false;
            cv::resize(giFrame, giFrame, genshin_handle.size_frame);

            out_genshin_screen.rect_client = cv::Rect(giRect.left, giRect.top, giRectClient.right - giRectClient.left, giRectClient.bottom - giRectClient.top);
            preprocess_raw_frame(giFrame);
            init_screen_frames(out_genshin_screen);
        }
        return true;
    }

    void init_screen_frames(GenshinScreen& out_genshin_screen)
    {
        auto& giFrame = out_genshin_screen.img_screen;
        int x = giFrame.cols;
        int y = giFrame.rows;

        // 小地图标定可能性区域计算参数
        int icon_sight_mayArea_left = static_cast<int>(x * 0.08);
        int icon_sight_mayArea_top = 0;
        int icon_sight_mayArea_width = static_cast<int>(x * 0.10);
        int icon_sight_mayArea_height = static_cast<int>(y * 0.10);
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
        int miniMap_mayArea_width = static_cast<int>(x * 0.18);
        int miniMap_mayArea_height = static_cast<int>(y * 0.22);
        // 小地图可能性区域
        cv::Rect Area_MiniMap_mayArea(
            miniMap_mayArea_left,
            miniMap_mayArea_top,
            miniMap_mayArea_width,
            miniMap_mayArea_height);
        out_genshin_screen.rects.minimap_maybe = Area_MiniMap_mayArea;

        // UID可能性区域计算参数
        int UID_mayArea_left = static_cast<int>(x * 0.88);
        int UID_mayArea_top = static_cast<int>(y * 0.97);
        int UID_mayArea_width = x - UID_mayArea_left;
        int UID_mayArea_height = y - UID_mayArea_top;
        // UID可能性区域
        cv::Rect Area_UID_mayArea(
            UID_mayArea_left,
            UID_mayArea_top,
            UID_mayArea_width,
            UID_mayArea_height);
        out_genshin_screen.rects.uid_maybe = Area_UID_mayArea;

        int UID_Rect_x = cvCeil(x - x * (1.0 - 0.865));
        int UID_Rect_y = cvCeil(y - 1080.0 * (1.0 - 0.9755));
        int UID_Rect_w = cvCeil(1920 * 0.11);
        int UID_Rect_h = cvCeil(1920 * 0.0938 * 0.11);
        out_genshin_screen.rects.uid = cv::Rect(UID_Rect_x, UID_Rect_y, UID_Rect_w, UID_Rect_h);

        // 获取maybe区域
        out_genshin_screen.imgs.icon_sight_maybe = giFrame(out_genshin_screen.rects.icon_sight_maybe);
        out_genshin_screen.imgs.minimap_maybe = giFrame(out_genshin_screen.rects.minimap_maybe);
        out_genshin_screen.imgs.uid_maybe = giFrame(out_genshin_screen.rects.uid_maybe);
        out_genshin_screen.imgs.uid = giFrame(out_genshin_screen.rects.uid);
    }

    void preprocess_raw_frame(cv::Mat& mat)
    {
        //四通道强制三通道
        if (mat.channels() == 4)
        {
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
        }

        //目前主要工作是提亮，因为画面中有标准白，不是问题，但没有标准黑是个问题
        double min_col, max_col;
        cv::Mat grayscale_mat;
        cv::cvtColor(mat, grayscale_mat, cv::COLOR_RGB2GRAY);
        cv::minMaxLoc(grayscale_mat, &min_col, &max_col);
        if (max_col > 250)
        {
            return;
        }
        cv::Mat f_mat;
        mat.convertTo(f_mat, CV_32FC3);
        f_mat = f_mat * (255.0 / max_col);
        f_mat.convertTo(mat, CV_8UC3);
        return;
    }
}