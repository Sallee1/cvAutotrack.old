#pragma once

namespace TianLi::Utils
{
    struct MatchKeyPoint
    {
        cv::Point2d query;
        cv::Point2d train;
    };

    /**
     * @brief 获取指定坐标附近的地图
     * @param map 地图图像
     * @param pos 坐标，如果给定的坐标无法获取，则会修改成最接近的有效坐标
     * @param size_r 半径
     * @return 图像的切片
     */
    cv::Mat get_some_map(const cv::Mat& map, cv::Point& pos, int size_r);

    double dis(cv::Point2d p);
    bool SPC(std::vector<double> lisx, std::vector<double> lisy, cv::Point2d& out);

    double stdev(std::vector<double> list);
    double stdev(std::vector<cv::Point2d> list);
    double stdev_abs(std::vector<double> list);

    cv::Mat crop_border(const cv::Mat& mat, double border);

    std::vector<double> extract_valid(std::vector<double> list);
    std::vector<cv::Point2d> extract_valid(std::vector<cv::Point2d> list);

    void remove_invalid(std::vector<cv::Point2f>& scene_pt, std::vector<cv::Point2f>& object_pt, double scale, std::vector<double>& x_list, std::vector<double>& y_list);

    int getMaxID(double lis[], int len);
    int getMinID(double lis[], int len);

    std::vector<cv::Point2f> Vector2UnitVector(std::vector<cv::Point2f> pLis);
    double Line2Angle(cv::Point2f p);
    cv::Point2d TransferAxes(cv::Point2d pos, cv::Point2d origin, double scale);
    cv::Point2d TransferAxes_inv(cv::Point2d pos, cv::Point2d origin, double scale);
    std::pair<cv::Point2d, int> ConvertSpecialMapsPosition(double x, double y);

    void draw_good_matches(const cv::Mat& img_scene, std::vector<cv::KeyPoint> keypoint_scene, const cv::Mat& img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<cv::DMatch>& good_matches);

    void calc_good_matches(const cv::Mat& img_scene, std::vector<cv::KeyPoint> keypoint_scene, const cv::Mat& img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<std::vector<cv::DMatch>>& KNN_m, double ratio_thresh, std::vector<cv::Point2f>& scene_goodmatch, std::vector<cv::Point2f>& object_goodmatch);

    bool getRegValue_REG_SZ(HKEY root, std::wstring item, std::wstring key, std::string& ret, int max_length);

    bool getRegValue_DWORD(HKEY root, std::wstring item, std::wstring key, int& ret);
}