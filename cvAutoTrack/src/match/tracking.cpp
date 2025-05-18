#include "pch.h"
#include "tracking.h"
#include "match/type/MatchType.h"
#include "resources/Resources.h"
#include "utils/Utils.h"

void Tracking::setMap(cv::Mat gi_map)
{
    _mapMat = gi_map;
}

void Tracking::setMiniMap(cv::Mat miniMapMat)
{
    _miniMapMat = miniMapMat;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher)
{
    if (isInit)return true;
    m_matcher->detect_and_compute(_mapMat, map.keypoints, map.descriptors);
    if (matcher == nullptr)
    {
        return false;
    }
    m_matcher = matcher;
    isInit = true;
    return true;
}

bool Tracking::Init(const std::shared_ptr<IMatcher>& matcher, std::vector<cv::KeyPoint>&& gi_map_keypoints, cv::Mat&& gi_map_descriptors)
{
    if (isInit)return true;
    map.keypoints = std::move(gi_map_keypoints);
    map.descriptors = std::move(gi_map_descriptors);
    if (matcher == nullptr)
    {
        return false;
    }
    m_matcher = matcher;
    isInit = true;
    return true;
}

void Tracking::UnInit()
{
    if (!isInit)return;
    _mapMat.release();
    _mapMat = cv::Mat();
    map.keypoints.clear();
    map.descriptors.release();
    isInit = false;
}

void Tracking::match()
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
    //pos = match_no_continuity(calc_continuity_is_faile);
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
        {
            isContinuity = true;
        }

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
            isContinuity = false;
            continuity_retry = 0;
        }
    }
}

cv::Point2d Tracking::match_continuity(bool& calc_continuity_is_faile)
{
    static cv::Mat img_scene(_mapMat);
    int real_some_map_size_r = DEFAULT_SOME_MAP_SIZE_R;

    cv::Point2d pos_object;

    //cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
    cv::Mat img_object = _miniMapMat;
    //不在城镇中时
    cv::Point some_map_center_pos = pos;
    cv::Mat someMap = TianLi::Utils::get_some_map(img_scene, some_map_center_pos, DEFAULT_SOME_MAP_SIZE_R);
    cv::Mat miniMap(img_object);
    cv::Mat miniMap_scale = img_object.clone();

    m_matcher->detect_and_compute(someMap, some_map);
    m_matcher->detect_and_compute(miniMap_scale, mini_map);

    // 如果搜索范围内可识别特征点数量少于2，则认为计算失败
    if (some_map.size() <= 2 || mini_map.size() <= 2)
    {
        calc_continuity_is_faile = true;
        return pos_object;
    }

    cv::Point2d p = match_impl(someMap, some_map, img_object, mini_map, calc_continuity_is_faile);
    if (calc_continuity_is_faile)
    {
        return {};
    }

    pos_object = cv::Point2d(p.x + some_map_center_pos.x - real_some_map_size_r, p.y + some_map_center_pos.y - real_some_map_size_r);

    double last_distance = std::sqrt(std::pow(static_cast<double>(pos_object.x) - last_pos.x, 2) +
        std::pow(static_cast<double>(pos_object.y) - last_pos.y, 2));
    if (!isMatchAllMap && last_pos.x != 0 && last_pos.y != 0 && last_distance > 200)
    {
        calc_continuity_is_faile = true;
        return {};
    }

    return pos_object;
}

/// <summary>
/// 匹配 不连续 全局匹配
/// </summary>
/// <param name="calc_is_faile"></param>
/// <returns></returns>
cv::Point2d Tracking::match_no_continuity(bool& calc_is_faile)
{
    //cv::Mat img_object = TianLi::Utils::crop_border(_miniMapMat, 0.15);
    cv::Mat img_object = _miniMapMat;
    m_matcher->detect_and_compute(img_object, mini_map);
    if (mini_map.size() <= 2)
    {
        calc_is_faile = true;
        return {};
    }
    return match_impl(_mapMat, map, img_object, mini_map, calc_is_faile);
}

cv::Point2d Tracking::match_impl(const cv::Mat& img_scene, const IMatcher::KeyMatPoint& keypoint_scene, const cv::Mat& img_object, const IMatcher::KeyMatPoint& keypoint_object, bool& calc_is_faile)
{
    cv::Point2d all_map_pos;
    // 没有提取到特征点直接返回，结果无效
    if (keypoint_object.keypoints.size() == 0)
    {
        calc_is_faile = true;
        return all_map_pos;
    }
    // 匹配特征点
    std::vector<std::vector<cv::DMatch>> KNN_m = m_matcher->match(keypoint_object, keypoint_scene, isContinuity);
    // 绘制关键点
    //cv::Mat match_results;
    //cv::drawMatches(img_object, keypoint_object.keypoints, img_scene, keypoint_scene.keypoints, KNN_m, match_results, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<std::vector<char>>());

    std::vector<cv::Point2f> good_matched_scene;
    std::vector<cv::Point2f> good_matched_object;
    TianLi::Utils::calc_good_matches(img_scene, keypoint_scene.keypoints, img_object, keypoint_object.keypoints, KNN_m, LOWE_RATIO_THRESH, good_matched_scene, good_matched_object);

    //auto good_matched_count = good_matched_scene.size();

    if (good_matched_scene.size() < 6)
    {
        return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
    }

    cv::Mat H, mask;
    H = cv::estimateAffinePartial2D(cv::Mat(good_matched_object), cv::Mat(good_matched_scene), mask, cv::RANSAC);

    cv::Mat test_out(img_scene.size(), CV_8UC3);
    cv::warpAffine(img_object, test_out, H, img_scene.size());

    int accept_count = cv::countNonZero(mask);
    double H_scale = (H.ptr<double>(0)[0] + H.ptr<double>(1)[1]) / 2;
    double H_maxerr = 0.05;
    double H_min_scale = 0.7;

    if (accept_count < 3 || static_cast<double>(accept_count) / good_matched_scene.size() < 0.3)
    {
        //矩阵的置信度不高，使用旧版的筛选算法
        return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
    }
    else if (
        H_scale < H_min_scale ||
        abs(H.ptr<double>(0)[0] - H.ptr<double>(1)[1]) > H_scale * H_maxerr ||
        abs(H.ptr<double>(0)[1]) > H_scale * H_maxerr ||
        abs(H.ptr<double>(0)[1]) > H_scale * H_maxerr)
    {
        //矩阵与经验不符（无旋转，无缩放，无斜切）
        return cleanAndComputePos_Old(good_matched_scene, good_matched_object, calc_is_faile);
    }
    else
    {
        std::vector<cv::Point2f> out_pt{ cv::Point2f(img_object.cols / 2, img_object.rows / 2) };
        cv::transform(out_pt, out_pt, H);
        return out_pt[0];
    }
}

cv::Point2d Tracking::cleanAndComputePos_Old(std::vector<cv::Point2f>& good_matched_scene, std::vector<cv::Point2f>& good_matched_object, bool& calc_is_faile)
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

cv::Point2d Tracking::getLocalPos()
{
    return pos;
}

bool Tracking::getIsContinuity()
{
    return isContinuity;
}