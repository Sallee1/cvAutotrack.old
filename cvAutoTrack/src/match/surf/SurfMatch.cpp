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

    matcher.detect_and_compute(someMap, some_map);
    matcher.detect_and_compute(miniMap_scale, mini_map);

    // 如果搜索范围内可识别特征点数量少于2，则认为计算失败
    if (some_map.size() <= 2 || mini_map.size() <= 2)
    {
        calc_continuity_is_faile = true;
        return pos_not_on_city;
    }

    cv::Point2d p = match_impl(someMap, some_map, img_object, mini_map, calc_continuity_is_faile);

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
    cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
    matcher.detect_and_compute(img_object, mini_map);
    if (mini_map.size() <= 2)
    {
        calc_is_faile = true;
        return {};
    }
    return match_impl(_mapMat, map, img_object, mini_map, calc_is_faile);
}

cv::Point2d SurfMatch::match_impl(const cv::Mat& img_scene, const Match::KeyMatPoint& keypoint_scene, const cv::Mat& img_object, const Match::KeyMatPoint& keypoint_object, bool& calc_is_faile)
{
    cv::Point2d all_map_pos;
    // 没有提取到特征点直接返回，结果无效
    if (keypoint_object.keypoints.size() == 0)
    {
        calc_is_faile = true;
        return all_map_pos;
    }
    // 匹配特征点
    std::vector<std::vector<cv::DMatch>> KNN_m = matcher.match(keypoint_object, keypoint_scene);

    std::vector<cv::Point2f> good_matched_scene;
    std::vector<cv::Point2f> good_matched_object;
    TianLi::Utils::calc_good_matches(img_scene, keypoint_scene.keypoints, img_object, keypoint_object.keypoints, KNN_m, SURF_MATCH_RATIO_THRESH, good_matched_scene, good_matched_object);
    // 算法需求将good_matched_object做原点平移
    std::transform(good_matched_object.begin(), good_matched_object.end(), good_matched_object.begin(), [&img_object](cv::Point2f p) {
        return p - static_cast<cv::Point2f>(img_object.size()) / 2;
        });

    if (good_matched_scene.size() < 10)
    {
        return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
    }

    cv::Mat H, mask;
    H = cv::findHomography(cv::Mat(good_matched_object), cv::Mat(good_matched_scene), cv::RANSAC, 3.0, mask);

    int accept_count = cv::countNonZero(mask);
    if (accept_count < 6 || static_cast<double>(accept_count) / good_matched_scene.size() < 0.3)
    {
        //矩阵的置信度不高，使用旧版的筛选算法
        return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
    }
    else {
        std::vector<cv::Point2f> out_pt{ cv::Point2f(0, 0) };
        cv::perspectiveTransform(out_pt, out_pt, H);
        return out_pt[0];
    }
}

cv::Point2d SurfMatch::cleanAndComputePos_Old(std::vector<cv::Point2f>& good_matched_scene, std::vector<cv::Point2f>& good_matched_object, bool& calc_is_faile)
{
    cv::Point2d all_map_pos{};

    std::vector<double> lisx;
    std::vector<double> lisy;
    TianLi::Utils::remove_invalid(good_matched_scene, good_matched_object, MAP_BOTH_SCALE_RATE, lisx, lisy);

    std::vector<cv::Point2d> list_filter_kp;
    for (int i = 0; i < good_matched_scene.size(); i++)
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

std::vector<std::vector<cv::DMatch>> Match::match(const KeyMatPoint& query_key_mat_point, const KeyMatPoint& train_key_mat_point)
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