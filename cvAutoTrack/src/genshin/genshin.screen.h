#pragma once
#include "match/type/MatchType.h"

namespace TianLi::Genshin
{
    bool get_genshin_screen(const GenshinHandle& genshin_handle, GenshinScreen& out_genshin_screen,
                            GenshinMinimap* out_minimap = nullptr);
    cv::Mat tone_map_hdr_to_sdr(const cv::Mat& hdr_rgba, double white_point);
}
