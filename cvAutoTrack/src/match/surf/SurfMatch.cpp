#include "pch.h"
#include "SurfMatch.h"
#include "match/type/MatchType.h"
#include "resources/Resources.h"
#include "utils/Utils.h"

void SurfMatch::setMap(cv::Mat gi_map)
{
    _mapMat = gi_map;
}

void SurfMatch::setMiniMap(cv::Mat miniMapMat)
{
    _miniMapMat = miniMapMat;
}

void SurfMatch::Init()
{
    if (isInit)return;
    matcher.detect_and_compute(_mapMat, map.keypoints, map.descriptors);
    isInit = true;
}

void SurfMatch::Init(std::vector<cv::KeyPoint>& gi_map_keypoints, cv::Mat& gi_map_descriptors)
{
    if (isInit)return;
    map.keypoints = std::move(gi_map_keypoints);
    map.descriptors = std::move(gi_map_descriptors);
    isInit = true;
}

void SurfMatch::UnInit()
{
    if (!isInit)return;
    _mapMat.release();
    _mapMat = cv::Mat();
    map.keypoints.clear();
    map.descriptors.release();
    isInit = false;
}

void SurfMatch::match()
{
    bool calc_is_faile = false;
    is_success_match = false;

    // 非连续匹配，匹配整个大地图
    if (isMatchAllMap)
    {
        pos = match_no_continuity(calc_is_faile);

        // 没有有效结果，结束
        if (calc_is_faile)
        {
            pos = last_pos;
            is_success_match = false;
            return;
        }
        continuity_retry = max_continuity_retry - 1;		//全局检测后只局部检测一次
    }

    // 尝试连续匹配，匹配角色附近小范围区域
    bool calc_continuity_is_faile = false;
    pos = match_continuity(calc_continuity_is_faile);

    if (!calc_continuity_is_faile)
    {
        last_pos = pos;
        continuity_retry = 0;

        if (isMatchAllMap)
        {
            isContinuity = false;
            isMatchAllMap = false;
        }
        else
            isContinuity = true;

        is_success_match = true;
    }
    else
    {
        pos = last_pos;
        is_success_match = false;
        continuity_retry++;

        if (continuity_retry >= max_continuity_retry)
        {
            isMatchAllMap = true;
            continuity_retry = 0;
        }
    }
}

cv::Point2d SurfMatch::match_continuity(bool& calc_continuity_is_faile)
{
    static cv::Mat img_scene(_mapMat);
    const auto minimap_scale_param = 1.0;
    int real_some_map_size_r = DEFAULT_SOME_MAP_SIZE_R;

    cv::Point2d pos_not_on_city;

    cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
    //不在城镇中时
    cv::Point some_map_center_pos = pos;
    cv::Mat someMap = TianLi::Utils::get_some_map(img_scene, some_map_center_pos, DEFAULT_SOME_MAP_SIZE_R);
    cv::Mat miniMap(img_object);
    cv::Mat miniMap_scale = img_object.clone();

    cv::resize(miniMap_scale, miniMap_scale, cv::Size(0, 0), minimap_scale_param, minimap_scale_param, cv::INTER_CUBIC);

    matcher.detect_and_compute(someMap, some_map);
    matcher.detect_and_compute(miniMap_scale, mini_map);

    // 如果搜索范围内可识别特征点数量少于2，则认为计算失败
    if (some_map.size() <= 2 || mini_map.size() <= 2)
    {
        calc_continuity_is_faile = true;
        return pos_not_on_city;
    }

    std::vector<std::vector<cv::DMatch>> KNN = matcher.match(mini_map, some_map);

    std::vector<TianLi::Utils::MatchKeyPoint> keypoint_matched;
    TianLi::Utils::calc_good_matches(someMap, some_map.keypoints, miniMap_scale, mini_map.keypoints, KNN, SURF_MATCH_RATIO_THRESH, keypoint_matched);

    std::vector<double> lisx;
    std::vector<double> lisy;
    TianLi::Utils::remove_invalid(keypoint_matched, MAP_BOTH_SCALE_RATE / minimap_scale_param, lisx, lisy);

    std::vector<cv::Point2d> keypoints_filtered;
    for (int i = 0; i < keypoint_matched.size(); i++)
    {
        keypoints_filtered.push_back(cv::Point2d(lisx[i], lisy[i]));
    }
    keypoints_filtered = TianLi::Utils::extract_valid(keypoints_filtered);

    lisx.clear();
    lisy.clear();
    for (int i = 0; i < keypoints_filtered.size(); i++)
    {
        lisx.push_back(keypoints_filtered[i].x);
        lisy.push_back(keypoints_filtered[i].y);
    }

    cv::Point2d p;
    if (!TianLi::Utils::SPC(lisx, lisy, p))
    {
        calc_continuity_is_faile = true;
        return pos_not_on_city;
    }
    pos_not_on_city = cv::Point2d(p.x + some_map_center_pos.x - real_some_map_size_r, p.y + some_map_center_pos.y - real_some_map_size_r);
    return pos_not_on_city;
}

/// <summary>
/// 匹配 不连续 全局匹配
/// </summary>
/// <param name="calc_is_faile"></param>
/// <returns></returns>
cv::Point2d SurfMatch::match_no_continuity(bool& calc_is_faile)
{
    cv::Point2d all_map_pos;

    cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);

    // 小地图区域计算特征点
    matcher.detect_and_compute(img_object, mini_map);
    // 没有提取到特征点直接返回，结果无效
    if (mini_map.keypoints.size() == 0)
    {
        calc_is_faile = true;
        return all_map_pos;
    }
    // 匹配特征点
    std::vector<std::vector<cv::DMatch>> KNN_m = matcher.match(mini_map, map);

    std::vector<TianLi::Utils::MatchKeyPoint> keypoint_list;
    TianLi::Utils::calc_good_matches(_mapMat, map.keypoints, img_object, mini_map.keypoints, KNN_m, SURF_MATCH_RATIO_THRESH, keypoint_list);

    std::vector<double> lisx;
    std::vector<double> lisy;
    TianLi::Utils::remove_invalid(keypoint_list, MAP_BOTH_SCALE_RATE, lisx, lisy);

    std::vector<cv::Point2d> list_filter_kp;
    for (int i = 0; i < keypoint_list.size(); i++)
    {
        list_filter_kp.push_back(cv::Point2d(lisx[i], lisy[i]));
    }
    list_filter_kp = TianLi::Utils::extract_valid(list_filter_kp);

    lisx.clear();
    lisy.clear();
    for (int i = 0; i < list_filter_kp.size(); i++)
    {
        lisx.push_back(list_filter_kp[i].x);
        lisy.push_back(list_filter_kp[i].y);
    }

    // 没有最佳匹配结果直接返回，结果无效
    if (std::min(lisx.size(), lisy.size()) == 0)
    {
        calc_is_faile = true;
        return all_map_pos;
    }
    // 从最佳匹配结果中剔除异常点计算角色位置返回
    if (!TianLi::Utils::SPC(lisx, lisy, all_map_pos))
    {
        calc_is_faile = true;
        return all_map_pos;
    }
    return all_map_pos;
}

cv::Point2d SurfMatch::getLocalPos()
{
    return pos;
}

bool SurfMatch::getIsContinuity()
{
    return isContinuity;
}

Match::Match(double hessian_threshold, int octaves, int octave_layers, bool extended, bool upright)
{
    detector = cv::xfeatures2d::SURF::create(hessian_threshold, octaves, octave_layers, extended, upright);
    //matcher  = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

std::vector<std::vector<cv::DMatch>> Match::match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors)
{
    std::vector<std::vector<cv::DMatch>> match_group;
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(query_descriptors, train_descriptors, match_group, 2);
    return match_group;
}

std::vector<std::vector<cv::DMatch>> Match::match(KeyMatPoint& query_key_mat_point, KeyMatPoint& train_key_mat_point)
{
    return match(query_key_mat_point.descriptors, train_key_mat_point.descriptors);
}

bool Match::detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    if (img.empty()) return  false;
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    if (keypoints.size() == 0) return false;
    return true;
}

bool Match::detect_and_compute(const cv::Mat& img, Match::KeyMatPoint& key_mat_point)
{
    return detect_and_compute(img, key_mat_point.keypoints, key_mat_point.descriptors);
}