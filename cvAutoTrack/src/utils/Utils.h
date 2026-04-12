#pragma once
#include <vector>
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
	 * @param pos 坐标，如果给定的坐标无法获取，则会修改成最接近的有效坐标
	 * @param size_r 半径
	 * @return 图像的roi
	 */
	cv::Rect2i get_rect_by_center_r(cv::Point& pos, int size_r);
	
	/**
	 * @brief 根据顶点获取包含所有顶点的固定大小矩形集，默认从少到多排列，用于局部匹配探索
	 * @param pts 点集
	 * @param size 输出矩形的大小
	 * @return 矩形集合
	 */
	std::vector<cv::Rect2i> getRectsByPoints(const std::vector<cv::Point2f>& pts, const cv::Size2i& size);

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

	std::vector<cv::Point2d> std_mean_filter(std::vector<cv::Point2d> list);
	std::vector<cv::Point2d> max_near_fliter(std::vector<cv::Point2d> list, double max_dist);

	int getMaxID(double lis[], int len);
	int getMinID(double lis[], int len);

	std::vector<cv::Point2f> Vector2UnitVector(std::vector<cv::Point2f> pLis);
	double Line2Angle(cv::Point2f p);
	cv::Point2d TransferAxes(cv::Point2d pos, cv::Point2d origin, double scale);
	cv::Point2d TransferAxes_inv(cv::Point2d pos, cv::Point2d origin, double scale);

	/**
	 * @brief 转换特殊地图的坐标
	 * @param x x坐标
	 * @param y y坐标
	 * @return 转换后的坐标和地图id，id为0表示普通地图
	 */
	std::pair<cv::Point2d, int> ConvertSpecialMapsPosition(double x, double y);

	void draw_good_matches(const cv::Mat& img_scene, std::vector<cv::KeyPoint> keypoint_scene, const cv::Mat& img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<cv::DMatch>& good_matches);

	void lowe_test(std::vector<std::vector<cv::DMatch>>& KNN_m, double ratio_thresh, std::vector<cv::DMatch>& out_good_matches);

	/**
	 * @brief 将dmatch转换为对应的点坐标
	 * @param keypoints_scene 场景关键点
	 * @param keypoints_object 小地图关键点
	 * @param good_matches 匹配结果
	 * @param scene_points 输出场景点
	 * @param object_points 输出小地图点
	 */
	void dmatch2cvPoints(const std::vector<cv::KeyPoint>& keypoints_scene, const std::vector<cv::KeyPoint>& keypoints_object, const std::vector<cv::DMatch>& good_matches, std::vector<cv::Point2f>& scene_points, std::vector<cv::Point2f>& object_points);

	bool getRegValue_REG_SZ(HKEY root, std::wstring item, std::wstring key, std::string& ret, int max_length);

	bool getRegValue_DWORD(HKEY root, std::wstring item, std::wstring key, int& ret);
}