#pragma once

namespace TianLi::Utils
{
    inline const std::map<std::string, std::pair<cv::Rect2i, cv::Rect2i>> map_mappers{
        //原图大小7200x7200，裁切范围(xywh)为[1200,600,4400,4400]
        { "希穆兰卡", { cv::Rect2i(0, 0, 4400, 4400),cv::Rect2i(533,267,1956,1956)} },
    };

    inline const std::map<std::string, std::pair<cv::Rect2i, cv::Rect2i>> area_mappers{
        { "希穆兰卡2x", { cv::Rect2i(4400, 0, 2933, 2933),cv::Rect2i(0, 0, 4400, 4400)} },
        { "希穆兰卡1x", { cv::Rect2i(4400, 2933, 1467, 1467),cv::Rect2i(0, 0, 4400, 4400)} },
    };
}