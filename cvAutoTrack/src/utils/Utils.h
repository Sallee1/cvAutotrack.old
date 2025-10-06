#pragma once
#include <match/IMatcher.h>

namespace TianLi::Utils
{
	struct MatchKeyPoint
	{
		cv::Point2d query;
		cv::Point2d train;
	};

	/**
	 * @brief 获取指定坐标附近的roi
	 * @param map 地图图像
	 * @param pos 坐标，如果给定的坐标无法获取，则会修改成最接近的有效坐标
	 * @param size_r 半径
	 * @return 图像的roi
	 */
	cv::Rect2i get_some_map_rect(const cv::Mat& map, cv::Point& pos, int size_r);

	double dis(cv::Point2d p);
	bool SPC(std::vector<double> lisx, std::vector<double> lisy, cv::Point2d& out);

	double stdev(std::vector<double> list);
	double stdev(std::vector<cv::Point2d> list);
	double stdev_abs(std::vector<double> list);

	cv::Mat crop_border(const cv::Mat& mat, double border);

	/**
	 * @brief 移除小地图的假特征点（从边缘生成），小地图是圆形的，且在图像中心
	 * @param input_img_size 输入的图像尺寸
	 * @param diameter 小地图的直径
	 * @param kp 输入的关键点
	 * @return 移除后的关键点
	 */
	IMatcher::KeyMatPoint remove_minimap_fake_keypoint(const cv::Size2i& input_img_size, float diameter, const IMatcher::KeyMatPoint& keypoints);

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