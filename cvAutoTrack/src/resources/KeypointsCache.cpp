#include <pch.h>
#include "KeypointsCache.h"
#include <fstream>
#include <filesystem>
#include "version/Version.h"
#include "serialize.h"
#include "resources/Resources.h"
#include "resources/map_mapper.h"

void MapKeypointCache::serialize(std::string outFileName)
{
	std::ofstream ofs(outFileName, std::fstream::out | std::fstream::binary);
	Tianli::Resources::Utils::serializeStream ss(ofs);
	ss << this->bulid_time;
	ss << this->bulid_version;
	ss << this->keyPoints;
	ss << this->descriptors;
	ss << this->bulid_version_end;
	ss.align();
	ofs.close();
}

void MapKeypointCache::deSerialize(std::string infileName, bool version_only)
{
	std::ifstream ifs(infileName, std::fstream::out | std::fstream::binary);
	Tianli::Resources::Utils::deSerializeStream dss(ifs);
	dss >> this->bulid_time;
	dss >> this->bulid_version;

	if (version_only)
	{
		ifs.close();
		return;
	}

	dss >> this->keyPoints;
	dss >> this->descriptors;
	dss >> this->bulid_version_end;
	ifs.close();
}

namespace {
	/**
	 * @brief 将输入的rect分块成子rect，且包含重叠区域
	 * @param input_rect 输入rect
	 * @param block_size 分块大小
	 * @param padding 重叠区域大小
	 * @return 返回值为pair的vector，first为包含重叠区域的子rect，second为不包含重叠区域的子rect
	 */
	std::vector<std::pair<cv::Rect2i, cv::Rect2i>> getRects(const cv::Rect2i& input_rect, const cv::Size2i& block_size, const cv::Size2i padding)
	{
		std::vector<std::pair<cv::Rect2i, cv::Rect2i>> rects;
		int x_blocks = (input_rect.width + block_size.width - 1) / block_size.width;
		int y_blocks = (input_rect.height + block_size.height - 1) / block_size.height;
		for (int y = 0; y < y_blocks; ++y)
		{
			for (int x = 0; x < x_blocks; ++x)
			{
				int x_start = input_rect.x + x * block_size.width;
				int y_start = input_rect.y + y * block_size.height;
				int x_end = std::min(x_start + block_size.width, input_rect.x + input_rect.width);
				int y_end = std::min(y_start + block_size.height, input_rect.y + input_rect.height);
				cv::Rect2i inner_rect(x_start, y_start, x_end - x_start, y_end - y_start);
				int padded_x_start = x_start - padding.width;
				int padded_y_start = y_start - padding.height;
				int padded_x_end = x_end + padding.width;
				int padded_y_end = y_end + padding.height;
				cv::Rect2i padded_rect(padded_x_start, padded_y_start, padded_x_end - padded_x_start, padded_y_end - padded_y_start);
				padded_rect &= input_rect; //确保不超出输入rect范围
				rects.emplace_back(padded_rect, inner_rect);
			}
		}
		return rects;
	}

	bool gen_map_keypoint_cache(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
	{
		auto& matcher = genshin_minimap.matcher;
		auto& map_template = Resources::getInstance().MapTemplate;

		//分块处理
		int block_size = 1200;
		int padding = 64; //64像素重叠
		auto rects = getRects(cv::Rect2i(0, 0, map_template.cols, map_template.rows), cv::Size2i(block_size, block_size), cv::Size2i(padding, padding));

		for (auto& rect : rects)
		{
			cv::Rect2i padded_rect = rect.first;
			cv::Rect2i inner_rect = rect.second;

			std::vector<cv::KeyPoint> kps;
			cv::Mat desc;

			//生成特征点
			cv::Mat roi_map_template = map_template(padded_rect);
			matcher->detect(roi_map_template, kps);

			//清除填充区域的关键点
			auto inner_rect_align = inner_rect;
			inner_rect_align.x -= padded_rect.x;
			inner_rect_align.y -= padded_rect.y;
			kps.erase(std::remove_if(kps.begin(), kps.end(), [&](const cv::KeyPoint& kp) {
				return inner_rect_align.contains(kp.pt) == false;
				}), kps.end());

			if (kps.empty())
				continue;

			//计算描述子
			matcher->compute(map_template(rect.first), kps, desc);

			//调整关键点坐标到全图坐标系
			for (auto& kp : kps)
			{
				kp.pt.x += rect.first.x;
				kp.pt.y += rect.first.y;
			}
			//合并关键点和描述子
			if (keypoints.empty())
			{
				keypoints = kps;
				descriptors = desc;
			}
			else
			{
				//合并描述子
				cv::vconcat(descriptors, desc, descriptors);
				//合并关键点
				keypoints.insert(keypoints.end(), kps.begin(), kps.end());
			}
		}
		return true;
	}

	/**
	 * @brief 使用映射表重映射关键点坐标
	 * @param keypoints 关键点
	 * @return 映射后的关键点
	 */
	void remap_keypoints(std::vector<cv::KeyPoint>& keypoints)
	{
		auto& layer_mapper = TianLi::Utils::layer_mapper;
		std::vector<cv::KeyPoint> remapped_keypoints;
		remapped_keypoints.reserve(keypoints.size());

		cv::Point2i center = Resources::getInstance().map_relative_center;
		cv::parallel_for_({ 0,static_cast<int>(keypoints.size()) },
			[&](const cv::Range& range)
			{
				for (int i = range.start; i < range.end; i++)
				{
					auto& kp = keypoints[i];
					auto& pt = kp.pt;
					//图层映射
					for (auto& [key, value] : layer_mapper)
					{
						auto srcRect = value.first + center;
						auto dstRect = value.second + center;

						if (srcRect.contains(pt))
						{
							pt = cv::Point2f{
								((float)dstRect.width / srcRect.width) * (pt.x - srcRect.x) + dstRect.x,
								((float)dstRect.height / srcRect.height) * (pt.y - srcRect.y) + dstRect.y };
							break;
						}
					}
				}
			}
		);
	}
}

bool save_map_keypoint_cache(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	gen_map_keypoint_cache(genshin_minimap, keypoints, descriptors);
	remap_keypoints(keypoints);
	std::string build_time = __DATE__ " " __TIME__;

	MapKeypointCache cache(
		build_time, TianLi::Version::build_version,
		keypoints, descriptors);
	std::filesystem::remove("cvAutoTrack_Cache.xml");
	cache.serialize("cvAutoTrack_Cache.xml");

	return true;
}

bool load_map_keypoint_cache(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (std::filesystem::exists("cvAutoTrack_Cache.xml") == false)
	{
		return false;
	}

	MapKeypointCache cache;
	try {
		cache.deSerialize("cvAutoTrack_Cache.xml", true);
	}
	catch (std::exception) {   //缓存损坏
		return false;
	}

	if (cache.bulid_version != TianLi::Version::build_version)    //版本不一致
		return false;

	cache.deSerialize("cvAutoTrack_Cache.xml");

	if (cache.bulid_version != cache.bulid_version_end)    //写入不完整
		return false;

	keypoints = cache.keyPoints;
	descriptors = cache.descriptors;
	return true;
}

bool get_map_keypoint(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (load_map_keypoint_cache(keypoints, descriptors) == false)
	{
		return save_map_keypoint_cache(genshin_minimap, keypoints, descriptors);
	}
	else
	{
		return true;
	}
}