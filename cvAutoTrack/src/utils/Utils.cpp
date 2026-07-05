#include "pch.h"
#include "Utils.h"
#include "resources/map_mapper_config.h"
#include <resources/Resources.h>
#include <resources/KeypointsCache.h>

namespace TianLi::Utils
{
	cv::Rect2i get_rect_by_center_r(cv::Point& pos, int size_r)
	{
		return cv::Rect2i(pos.x - size_r, pos.y - size_r, size_r + size_r, size_r + size_r);
	}
	double dis(cv::Point2d p)
	{
		return sqrt(p.x * p.x + p.y * p.y);
	}

	std::vector<cv::Point2d> std_mean_filter(std::vector<cv::Point2d> list)
	{
		std::vector<cv::Point2d> valid_list;

		if (list.size() <= 3)
		{
			return list;
		}

		std::vector<double> x_list;
		std::vector<double> y_list;
		for (auto& point : list)
		{
			x_list.push_back(point.x);
			y_list.push_back(point.y);
		}

		std::vector<double> x_valid_list;
		std::vector<double> y_valid_list;

		// double mean = std::accumulate(list.begin(), list.end(), 0.0) / list.size_r();
		double x_mean = std::accumulate(x_list.begin(), x_list.end(), 0.0) / x_list.size();
		double y_mean = std::accumulate(y_list.begin(), y_list.end(), 0.0) / y_list.size();

		double x_accum = 0.0;
		std::for_each(x_list.begin(), x_list.end(), [&](const double d)
			{ x_accum += (d - x_mean) * (d - x_mean); });
		double y_accum = 0.0;
		std::for_each(y_list.begin(), y_list.end(), [&](const double d)
			{ y_accum += (d - y_mean) * (d - y_mean); });

		double x_stdev = sqrt(x_accum / (x_list.size() - 1));
		double y_stdev = sqrt(y_accum / (y_list.size() - 1));

		double param = 1.0;
		if (list.size() > 100)
		{
			param = 0.382;
		}
		else if (list.size() > 50)
		{
			param = 0.618;
		}

		int valid_count = 0;
		for (auto& point : list)
		{
			if (abs(point.x - x_mean) < param * x_stdev && abs(point.y - y_mean) < param * y_stdev)
			{
				x_valid_list.push_back(point.x);
				y_valid_list.push_back(point.y);
				valid_count = valid_count + 1;
			}
		}

		for (int i = 0; i < valid_count; i++)
		{
			valid_list.push_back(cv::Point2d(x_valid_list[i], y_valid_list[i]));
		}
		return valid_list;
	}

	std::vector<cv::Point2d> max_near_fliter(std::vector<cv::Point2d> list,double max_dist)
	{
		//遍历每一个点，找到在附近的邻居点
		//建议只在输入小于100时使用
		int max_near_index = 0;
		int max_near_count = 0;
		int near_count = 0;
		std::vector<cv::Point2d> nears;
		for (int i = 0; i < list.size(); i++)
		{
			near_count = 0;
			std::vector<cv::Point2d> cur_nears;
			for (int j = 0; j < list.size(); j++)
			{
				double dist = sqrt((list[i].x - list[j].x) * (list[i].x - list[j].x) + (list[i].y - list[j].y) * (list[i].y - list[j].y));
				if (dist < max_dist)
				{
					cur_nears.emplace_back(list[j]);
					near_count++;
				}
			}
			if (near_count > max_near_count)
			{
				max_near_index = i;
				max_near_count = near_count;
				nears = std::move(cur_nears);
			}
		}
		return nears;
	}

	cv::Mat crop_border(const cv::Mat& mat, double border)
	{
		int crop_size = static_cast<int>((mat.rows + mat.cols) * 0.5 * border);
		return mat(cv::Rect(crop_size, crop_size, mat.cols - crop_size * 2, mat.rows - crop_size * 2));
	}

	IMatcher::KeyMatPoint remove_minimap_fake_keypoint(const cv::Size2i& input_img_size, float diameter, const IMatcher::KeyMatPoint& keypoints)
	{
		IMatcher::KeyMatPoint result;
		result.descriptors = cv::Mat(0, keypoints.descriptors.cols, keypoints.descriptors.type());
		size_t keypoint_size = keypoints.keypoints.size();
		if (keypoint_size == 0)
		{
			return result;
		}
		float radius = diameter / 2.0f;
		float radius_sq = radius * radius;
		cv::Point2f center(input_img_size.width / 2.0f, input_img_size.height / 2.0f);

		for (size_t i = 0; i < keypoint_size; ++i)
		{
			cv::Point2f pt = keypoints.keypoints[i].pt;
			float dx = pt.x - center.x;
			float dy = pt.y - center.y;
			float dist_sq = dx * dx + dy * dy;
			if (dist_sq <= radius_sq)
			{
				result.keypoints.push_back(keypoints.keypoints[i]);
				result.descriptors.push_back(keypoints.descriptors.row(static_cast<int>(i)));
			}
		}
		return result;
	}

	std::vector<cv::Rect2i> getRectsByPoints(const std::vector<cv::Point2f>& pts, const cv::Size2i& size) {
		//使用网格划分特征点
		using GridIndex = std::pair<int, int>;
		std::map<GridIndex, cv::Rect2i> grid_rect;
		std::map<GridIndex, int> grid_pt_count;
		for (auto& pt : pts)
		{
			GridIndex pt_index{ static_cast<int>(pt.x) / size.width,static_cast<int>(pt.y) / size.height };
			if (!grid_rect.contains(pt_index))
			{
				grid_rect[pt_index] = cv::Rect2i{ pt,cv::Size2i{1,1} };
				grid_pt_count[pt_index] = 1;
			}
			else {
				grid_rect[pt_index] |= cv::Rect2i{ pt,cv::Size2i{1,1} };
				grid_pt_count[pt_index] ++;
			}
		}
		//然后提取每一个单元格的矩形
		std::vector<std::pair<int, cv::Rect2i>> out_rects_with_count;
		out_rects_with_count.reserve(grid_rect.size());
		for (auto& [rect_key, rect] : grid_rect)
		{
			//规格化尺寸
			cv::Rect2i norm_rect{
				rect.width / 2 + rect.x - size.width / 2,
				rect.height / 2 + rect.y - size.height / 2,
				size.width,
				size.height
			};
			out_rects_with_count.emplace_back(std::make_pair(grid_pt_count[rect_key],norm_rect));
		}
		std::sort(out_rects_with_count.begin(), out_rects_with_count.end(),
			[&](const std::pair<int, cv::Rect2i>& l, std::pair<int, cv::Rect2i>& r) {
				return l.first >= r.first;
			});

		//整理为顶点数从大到小的列表
		std::vector<cv::Rect2i> out_rects;
		out_rects.reserve(out_rects_with_count.size());
		for (auto& [rect_key, rect] : out_rects_with_count)
		{
			out_rects.emplace_back(rect);
		}
		return out_rects;
	}


	bool SPC(std::vector<double> lisx, std::vector<double> lisy, cv::Point2d& out)
	{
		double meanx = std::accumulate(lisx.begin(), lisx.end(), 0.0) / lisx.size();
		double meany = std::accumulate(lisy.begin(), lisy.end(), 0.0) / lisy.size();
		double x = meanx;
		double y = meany;
		if (lisx.size() > 3 && lisy.size() > 3)
		{
			double accumx = 0.0;
			double accumy = 0.0;
			for (int i = 0; i < (lisx.size() > lisy.size() ? lisy.size() : lisx.size()); i++)
			{
				accumx += (lisx[i] - meanx) * (lisx[i] - meanx);
				accumy += (lisy[i] - meany) * (lisy[i] - meany);
			}

			double stdevx = sqrt(accumx / (lisx.size() - 1)); // 标准差
			double stdevy = sqrt(accumy / (lisy.size() - 1)); // 标准差

			double sumx = 0;
			double sumy = 0;
			double numx = 0;
			double numy = 0;
			for (int i = 0; i < (lisx.size() > lisy.size() ? lisy.size() : lisx.size()); i++)
			{
				if (abs(lisx[i] - meanx) < 1 * stdevx)
				{
					sumx += lisx[i];
					numx++;
				}

				if (abs(lisy[i] - meany) < 1 * stdevy)
				{
					sumy += lisy[i];
					numy++;
				}
			}
			x = sumx / numx;
			y = sumy / numy;
			out = cv::Point2d(x, y);
		}
		else
		{
			out = cv::Point2d();
			return 0;
		}
		return true;
	}

	int getMaxID(double lis[], int len)
	{
		int maxId = 0;
		for (int i = 1; i < len; i++)
		{
			if (lis[i] > lis[maxId])
			{
				maxId = i;
			}
		}
		return maxId;
	}

	int getMinID(double lis[], int len)
	{
		int maxId = 0;
		for (int i = 1; i < len; i++)
		{
			if (lis[i] < lis[maxId])
			{
				maxId = i;
			}
		}
		return maxId;
	}

	std::vector<cv::Point2f> Vector2UnitVector(std::vector<cv::Point2f> pLis)
	{
		double length = 1;
		std::vector<cv::Point2f> res;
		for (int i = 0; i < pLis.size(); i++)
		{
			length = sqrt(pLis[i].x * pLis[i].x + pLis[i].y * pLis[i].y);
			res.emplace_back(cv::Point2f((float)(pLis[i].x / length), (float)(pLis[i].y / length)));
		}
		return res;
	}

	double Line2Angle(cv::Point2f p)
	{
		const double rad2degScale = 180 / CV_PI;
		double res = atan2(-p.y, p.x) * rad2degScale;
		res = res - 90; // 从屏幕空间左侧水平线为0度转到竖直向上为0度
		if (res < -180.0)
			res = res + 360;
		return res;
	}

	cv::Point2d TransferAxes(cv::Point2d pos, cv::Point2d origin, double scale)
	{
		return cv::Point2d((pos - origin) * scale);
	}

	cv::Point2d TransferAxes_inv(cv::Point2d pos, cv::Point2d origin, double scale)
	{
		return cv::Point2d(pos / scale + origin);
	}

	std::pair<cv::Point2d, int> ConvertSpecialMapsPosition(double x, double y)
	{
		auto& mapper = TianLi::Resources::MapMapperManager::getInstance();
		if (mapper.isLoaded())
		{
			auto result = mapper.convertSpecialMap(x, y);
			if (result.second > 0)
				return result;
		}

		// 模块未加载时，保持原坐标
		return { cv::Point2d(x, y), 0 };
	}

	void draw_good_matches(const cv::Mat& img_scene, std::vector<cv::KeyPoint> keypoint_scene, const cv::Mat& img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<cv::DMatch>& good_matches)
	{
		cv::Mat img_matches, imgmap, imgminmap;
		drawKeypoints(img_scene, keypoint_scene, imgmap, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		drawKeypoints(img_object, keypoint_object, imgminmap, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		drawMatches(img_object, keypoint_object, img_scene, keypoint_scene, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}

	void lowe_test(std::vector<std::vector<cv::DMatch>>& KNN_m, double ratio_thresh, std::vector<cv::DMatch>& out_good_matches)
	{
		for (auto& m : KNN_m)
		{
			if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance)
			{
				out_good_matches.emplace_back(m[0]);
			}
		}
	}

	void dmatch2cvPoints(const std::vector < cv::KeyPoint>& keypoints_scene, const std::vector < cv::KeyPoint>& keypoints_object, const std::vector<cv::DMatch>& good_matches, std::vector<cv::Point2f>& scene_points, std::vector<cv::Point2f>& object_points)
	{
		scene_points.reserve(good_matches.size());
		object_points.reserve(good_matches.size());
		for (size_t i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			scene_points.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
			object_points.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		}
	}

	// 注册表读取
	bool getRegValue_REG_SZ(HKEY root, std::wstring item, std::wstring key, std::string& ret, int max_length)
	{
		HKEY hKey;
		long lRes = RegOpenKeyEx(root, item.c_str(), 0, KEY_READ, &hKey);
		if (lRes != ERROR_SUCCESS)
		{
			RegCloseKey(hKey);
			return false;
		}
		wchar_t* lpData = new wchar_t[max_length];
		DWORD dwType = REG_SZ;
		DWORD dwSize = max_length;

		lRes = RegGetValue(hKey, NULL, key.c_str(), RRF_RT_REG_SZ, &dwType, lpData, &dwSize);
		if (lRes != ERROR_SUCCESS)
		{
			RegCloseKey(hKey);
			delete[] lpData;
			return false;
		}

		char* lpDataA = new char[max_length];
		size_t lpDataALen;
		DWORD isSuccess;
		isSuccess = wcstombs_s(&lpDataALen, lpDataA, max_length, lpData, max_length - 1);
		if (isSuccess == ERROR_SUCCESS)
			ret = lpDataA;
		else
		{
			RegCloseKey(hKey);
			delete[] lpData;
			delete[] lpDataA;
			return false;
		}
		RegCloseKey(hKey);
		delete[] lpData;
		delete[] lpDataA;
		return true;
	}

	bool getRegValue_DWORD(HKEY root, std::wstring item, std::wstring key, int& ret)
	{
		HKEY hKey;
		long lRes = RegOpenKeyEx(root, item.c_str(), 0, KEY_READ, &hKey);
		if (lRes != ERROR_SUCCESS)
		{
			RegCloseKey(hKey);
			return false;
		}
		DWORD lpData;
		DWORD dwType = REG_DWORD;
		DWORD dwSize = sizeof(DWORD);

		lRes = RegGetValue(hKey, NULL, key.c_str(), RRF_RT_REG_DWORD, &dwType, &lpData, &dwSize);
		if (lRes != ERROR_SUCCESS)
		{
			RegCloseKey(hKey);
			return false;
		}

		ret = lpData;
		RegCloseKey(hKey);
		return true;
	}
}