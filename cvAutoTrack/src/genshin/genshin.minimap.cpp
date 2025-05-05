#include "pch.h"
#include "genshin.minimap.h"
#include "resources/Resources.h"

namespace TianLi::Genshin {
    class MinimapFounder {
        /**
         * @brief 检查当前的匹配
         * @param rect 待输出的举行
         * @param distance 匹配距离
         * @param genshin_icon_sight icon_sight对象
         * @param is_ctrl_mode 是否是控制器模式
         * @return 是否通过测试
         */
        bool check_match(const cv::Rect& rect, double distance, const GenshinIconSight& genshin_icon_sight, bool& is_handle) {
            if (rect.width < genshin_icon_sight.config.min_size || rect.height < genshin_icon_sight.config.min_size)
            {
                return false;   //roi太小
            }
            if (rect.width > genshin_icon_sight.config.max_size || rect.height > genshin_icon_sight.config.max_size)
            {
                return false;   //roi太大
            }
            if (distance > genshin_icon_sight.config.min_distance)
            {
                return false;   //匹配率不足
            }
            double ratio = (double)rect.width / (double)rect.height;
            if (ratio < 1.0) ratio = 1.0 / ratio;
            if (ratio > genshin_icon_sight.config.ratio || ratio < 1.0 / genshin_icon_sight.config.ratio)
            {
                return false;   //宽高比不符合
            }

            // 测试成功，判断是键鼠还是手柄
            if (rect.width < genshin_icon_sight.config.ctrl_size)
            {
                is_handle = true;
            }
            else
            {
                is_handle = false;
            }

            return true;
        }

        /**
         * @brief 待匹配的形状
         * @param shape_src 输入形状轮廓坐标
         * @param dst_img 待测试的图像
         * @param threshold 测试图像阈值
         * @param out_rect 待匹配形状的输出外框
         * @param out_distance
         * @return 是否查找成功
         */
        bool shape_match(cv::Mat shape_src, cv::Mat icon_sight_maybe, double threshold, GenshinIconSight& genshin_icon_sight)
        {
            cv::cvtColor(icon_sight_maybe, icon_sight_maybe, cv::COLOR_BGR2GRAY);
            double min_val, max_val;
            cv::minMaxLoc(icon_sight_maybe, &min_val, &max_val);
            icon_sight_maybe = icon_sight_maybe > static_cast<int>(max_val * threshold);
            cv::Mat icon_sight_maybe_inv = 255 - icon_sight_maybe;

            //查找轮廓
            std::vector<cv::Mat> contours_icon_sight_maybe;
            cv::findContours(icon_sight_maybe, contours_icon_sight_maybe,
                cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
            std::vector<cv::Mat> contours_icon_sight_maybe_inv;
            cv::findContours(icon_sight_maybe_inv, contours_icon_sight_maybe_inv,
                cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
            contours_icon_sight_maybe.insert(contours_icon_sight_maybe.end(), contours_icon_sight_maybe_inv.begin(),
                contours_icon_sight_maybe_inv.end());

            //粗形状匹配
            double min_distance = 1.0;
            int max_index = -1;
            bool is_ctrl_mode = false;
            for (int i = 0; i < contours_icon_sight_maybe.size(); i++)
            {
                cv::Mat debug_draw_contour_icon_sight_maybe = cv::Mat::zeros(icon_sight_maybe.size(), CV_8UC1);
                cv::drawContours(debug_draw_contour_icon_sight_maybe, contours_icon_sight_maybe, i, cv::Scalar(255), 1);

                double distance = cv::matchShapes(shape_src, contours_icon_sight_maybe[i], cv::CONTOURS_MATCH_I1, 0);
                if (distance < min_distance)
                {
                    bool is_ctrl_mode_maybe = false;
                    cv::Rect rect = cv::boundingRect(contours_icon_sight_maybe[i]);
                    if (!check_match(rect, distance, genshin_icon_sight, is_ctrl_mode_maybe))
                    {
                        continue;
                    }
                    is_ctrl_mode = is_ctrl_mode_maybe;
                    min_distance = distance;
                    max_index = i;
                }
            }
            if (max_index == -1)
            {
                return false;
            }

            genshin_icon_sight.is_visial = true;
            genshin_icon_sight.is_ctrl_mode = is_ctrl_mode;
            genshin_icon_sight.rect_Icon_sight = cv::boundingRect(contours_icon_sight_maybe[max_index]);
            return true;
        }

        bool match_icon_sight(const GenshinScreen& genshin_screen, GenshinIconSight& out_genshin_icon_sight)
        {
            static cv::Mat icon_sight_tpl;
            static std::vector<cv::Mat> contours_icon_sight_tpl;
            static bool is_first = true;

            // 从资源预处理模板
            if (is_first)
            {
                // 缩放键鼠和手柄模式的模板
                cv::Mat icon_sight = Resources::getInstance().IconSightTemplate;
                cv::cvtColor(icon_sight, icon_sight_tpl, cv::COLOR_BGR2GRAY);
                icon_sight_tpl = icon_sight_tpl > static_cast<int>(0.8 * 255.0);

                //查找轮廓
                cv::findContours(icon_sight_tpl, contours_icon_sight_tpl,
                    cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                cv::Mat debug_draw_contour_icon_sight = cv::Mat::zeros(icon_sight_tpl.size(), CV_8UC1);
                cv::drawContours(debug_draw_contour_icon_sight, contours_icon_sight_tpl, 0, cv::Scalar(255), 1);
                is_first = false;
            }

            cv::Mat icon_sight_maybe = genshin_screen.imgs.icon_sight_maybe.clone();
            //暗形状匹配（大部分情况）
            if (shape_match(contours_icon_sight_tpl[0], icon_sight_maybe,
                out_genshin_icon_sight.config.icon_sight_threshold_low,
                out_genshin_icon_sight))
            {
                return true;
            }
            //亮形状匹配
            if (shape_match(contours_icon_sight_tpl[0], icon_sight_maybe, out_genshin_icon_sight.config.icon_sight_threshold_high, out_genshin_icon_sight))
            {
                return true;
            }
            return false;
        }

        bool match_quest(const GenshinScreen& genshin_screen, const GenshinIconSight& genshin_icon_sight, cv::Size2i minimap_size, float tpl_threshold)
        {
            //准备模板
            static bool is_first = true;
            static cv::Mat img_icon_quest_tpl;
            static cv::Mat f_img_icon_quest_tpl;
            static cv::Mat mask_icon_quest_tpl;
            static cv::Mat img_icon_quest_tpl_ctrl;
            static cv::Mat f_img_icon_quest_tpl_ctrl;
            static cv::Mat mask_icon_quest_tpl_ctrl;
            if (is_first)
            {
                cv::Mat& raw_icon_quest_tpl = Resources::getInstance().IconQuestTemplate;
                cv::resize(raw_icon_quest_tpl, img_icon_quest_tpl, { genshin_screen.config.icon_size,genshin_screen.config.icon_size }, 0.0, 0.0, cv::INTER_AREA);
                cv::resize(raw_icon_quest_tpl, img_icon_quest_tpl_ctrl, { genshin_screen.config.icon_size_ctrl,genshin_screen.config.icon_size_ctrl }, 0.0, 0.0, cv::INTER_AREA);
                cv::cvtColor(img_icon_quest_tpl, img_icon_quest_tpl, cv::COLOR_RGB2GRAY);
                cv::cvtColor(img_icon_quest_tpl_ctrl, img_icon_quest_tpl_ctrl, cv::COLOR_RGB2GRAY);
                mask_icon_quest_tpl = img_icon_quest_tpl > 250;
                mask_icon_quest_tpl_ctrl = img_icon_quest_tpl_ctrl > 250;
                img_icon_quest_tpl.convertTo(f_img_icon_quest_tpl, CV_32FC1, (1.0 / 255.0));
                img_icon_quest_tpl_ctrl.convertTo(f_img_icon_quest_tpl_ctrl, CV_32FC1, (1.0 / 255.0));
                is_first = false;
            }

            //定位旗子的位置，辅助判断是否是小地图界面
            cv::Rect rect_Icon_sight_screen = genshin_icon_sight.rect_Icon_sight + genshin_screen.rects.icon_sight_maybe.tl();
            cv::Rect rect_Icon_quest_maybe;
            if (genshin_icon_sight.is_ctrl_mode)
            {
                rect_Icon_quest_maybe = cv::Rect(
                    cv::Point2i{
                        rect_Icon_sight_screen.x + rect_Icon_sight_screen.width / 2 - static_cast<int>(minimap_size.width * genshin_screen.config.controller_ui_scale + 26.5) ,
                        rect_Icon_sight_screen.y + static_cast<int>(minimap_size.height * genshin_screen.config.controller_ui_scale - 53)
                    },
                    cv::Size2i{
                        53,53
                    }
                );
            }
            else
            {
                rect_Icon_quest_maybe = cv::Rect(
                    cv::Point2i{
                        rect_Icon_sight_screen.x + rect_Icon_sight_screen.width / 2 - minimap_size.width - 32 ,
                        rect_Icon_sight_screen.y + minimap_size.height - 64
                    },
                    cv::Size2i{
                        64,64
                    }
                );
            }

            if (rect_Icon_quest_maybe.x < 0 || rect_Icon_quest_maybe.y < 0)
            {
                //假阳性结果可能会导致越界
                return false;
            }

            cv::Mat img_icon_quest_maybe = genshin_screen.img_screen(rect_Icon_quest_maybe).clone();
            //cv::imshow("debug_icon_quest_maybe", img_icon_quest_maybe);
            //cv::waitKey(1);

            //简单做一个亮度规格化，然后执行模板匹配
            cv::cvtColor(img_icon_quest_maybe, img_icon_quest_maybe, cv::COLOR_RGB2GRAY);

            cv::Mat f_img_icon_quest_maybe;
            img_icon_quest_maybe.convertTo(f_img_icon_quest_maybe, CV_32FC1);

            double min_val, max_val;
            cv::minMaxLoc(f_img_icon_quest_maybe, &min_val, &max_val);
            f_img_icon_quest_maybe = f_img_icon_quest_maybe / max_val;
            f_img_icon_quest_maybe.convertTo(img_icon_quest_maybe, CV_8U, 255.0);

            if (max_val < 230)
            {
                //应该没有玩家用这么低的亮度玩原吧...算匹配失败
                //不然纯色底不好处理
                return false;
            }

            //模板匹配
            cv::Mat match_result;
            cv::Mat* tpl_img = &img_icon_quest_tpl;
            cv::Mat* tpl_mask = &mask_icon_quest_tpl;
            cv::Mat* f_tpl_img = &f_img_icon_quest_tpl;

            if (genshin_icon_sight.is_ctrl_mode)
            {
                tpl_img = &img_icon_quest_tpl_ctrl;
                tpl_mask = &mask_icon_quest_tpl_ctrl;
                f_tpl_img = &f_img_icon_quest_tpl_ctrl;
            }
            cv::matchTemplate(img_icon_quest_maybe, *tpl_img, match_result, cv::TM_SQDIFF, *tpl_mask);

            //根据模板粗匹配结果，计算平均差值
            cv::minMaxLoc(img_icon_quest_maybe, &min_val, &max_val);
            cv::Point2i min_loc, max_loc;
            cv::minMaxLoc(match_result, &min_val, &max_val, &min_loc, &max_loc);
            cv::Mat diff;
            cv::subtract(f_img_icon_quest_maybe(cv::Rect{ min_loc ,f_tpl_img->size() }), *f_tpl_img, diff);
            diff = cv::abs(diff);

            double mean = cv::mean(diff, *tpl_mask)[0];

            if (mean > tpl_threshold)
            {
                return false;
            }
            return true;
        }

    public:
        bool cailb_minimap_impl(const GenshinScreen& genshin_screen, GenshinMinimap& out_genshin_minimap)
        {
            // rect
            cv::Rect minimap_rect = genshin_screen.rects.minimap;
            if (genshin_screen.config.is_search_mode)
            {
                static GenshinIconSight genshin_icon_sight;
                if (match_icon_sight(genshin_screen, genshin_icon_sight))
                {
                    if (genshin_icon_sight.is_ctrl_mode)
                    {
                        genshin_icon_sight.rect_Icon_sight = cv::Rect{
                            cv::Point2i{genshin_icon_sight.rect_Icon_sight.x + genshin_icon_sight.rect_Icon_sight.width / 2 - genshin_screen.config.icon_size_ctrl / 2,
                            genshin_icon_sight.rect_Icon_sight.y + genshin_icon_sight.rect_Icon_sight.height / 2 - genshin_screen.config.icon_size_ctrl / 2},
                        cv::Size2i{genshin_screen.config.icon_size_ctrl, genshin_screen.config.icon_size_ctrl} };
                    }
                    else
                    {
                        genshin_icon_sight.rect_Icon_sight = cv::Rect{
                        genshin_icon_sight.rect_Icon_sight.x + genshin_icon_sight.rect_Icon_sight.width / 2 - genshin_screen.config.icon_size / 2,
                        genshin_icon_sight.rect_Icon_sight.y + genshin_icon_sight.rect_Icon_sight.height / 2 - genshin_screen.config.icon_size / 2,
                        genshin_screen.config.icon_size, genshin_screen.config.icon_size };
                    }
                    //cv::Mat debug_icon_sight_found = genshin_screen.imgs.icon_sight_maybe(genshin_icon_sight.rect_Icon_sight);
                    //cv::imshow("debug_icon_sight_found", debug_icon_sight_found);
                    //cv::waitKey(1);
                }

                if (!genshin_icon_sight.is_visial)
                {
                    return false;
                }
                //检查旗子
                if (!match_quest(genshin_screen, genshin_icon_sight, out_genshin_minimap.config.minimap_size, genshin_icon_sight.config.tplmatch_max_diff))
                {
                    return false;
                }
                //旗子检查通过，有小地图，则给小地图相关参数赋值
                cv::Rect rect_Icon_sight_screen = genshin_icon_sight.rect_Icon_sight + genshin_screen.rects.icon_sight_maybe.tl();
                if (genshin_icon_sight.is_ctrl_mode)
                {
                    minimap_rect.x = rect_Icon_sight_screen.x + rect_Icon_sight_screen.width / 2 - static_cast<int>(out_genshin_minimap.config.minimap_size.width * genshin_screen.config.controller_ui_scale);
                    minimap_rect.y = rect_Icon_sight_screen.y;
                    minimap_rect.width = static_cast<int>(out_genshin_minimap.config.minimap_size.width * genshin_screen.config.controller_ui_scale);
                    minimap_rect.height = static_cast<int>(out_genshin_minimap.config.minimap_size.height * genshin_screen.config.controller_ui_scale);
                }
                else
                {
                    minimap_rect.x = rect_Icon_sight_screen.x + rect_Icon_sight_screen.width / 2 - static_cast<int>(out_genshin_minimap.config.minimap_size.width);
                    minimap_rect.y = rect_Icon_sight_screen.y;
                    minimap_rect.width = static_cast<int>(out_genshin_minimap.config.minimap_size.width);
                    minimap_rect.height = static_cast<int>(out_genshin_minimap.config.minimap_size.height);
                }
            }
            // center point
            auto minimap_center = cv::Point(minimap_rect.x + (minimap_rect.width) / 2, minimap_rect.y + (minimap_rect.height) / 2);
            out_genshin_minimap.img_minimap = genshin_screen.img_screen(minimap_rect);
            out_genshin_minimap.rect_minimap = minimap_rect;
            out_genshin_minimap.point_minimap_center = minimap_center;

            int Avatar_Rect_x = cvRound(minimap_rect.width * 0.4);
            int Avatar_Rect_y = cvRound(minimap_rect.height * 0.4);
            int Avatar_Rect_w = cvRound(minimap_rect.width * 0.2);
            int Avatar_Rect_h = cvRound(minimap_rect.height * 0.2);

            out_genshin_minimap.rect_avatar = cv::Rect(Avatar_Rect_x, Avatar_Rect_y, Avatar_Rect_w, Avatar_Rect_h);
            out_genshin_minimap.img_avatar = out_genshin_minimap.img_minimap(out_genshin_minimap.rect_avatar);

            int Viewer_Rect_x = cvRound(minimap_rect.width * 0.2);
            int Viewer_Rect_y = cvRound(minimap_rect.height * 0.2);
            int Viewer_Rect_w = cvRound(minimap_rect.width * 0.6);
            int Viewer_Rect_h = cvRound(minimap_rect.height * 0.6);

            out_genshin_minimap.rect_viewer = cv::Rect(Viewer_Rect_x, Viewer_Rect_y, Viewer_Rect_w, Viewer_Rect_h);
            out_genshin_minimap.img_viewer = out_genshin_minimap.img_minimap(out_genshin_minimap.rect_viewer);

            return true;
        }
    };

    bool find_minimap(const GenshinScreen& genshin_screen, GenshinMinimap& out_genshin_minimap)
    {
        static MinimapFounder founder;
        return founder.cailb_minimap_impl(genshin_screen, out_genshin_minimap);
    }
}