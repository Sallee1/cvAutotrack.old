#include <pch.h>
#include "KeypointsCache.h"
#include <fstream>
#include <filesystem>
#include "version/Version.h"
#include "serialize.h"
#include "resources/Resources.h"

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
	}
}

bool save_map_keypoint_cache(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	auto& matcher = genshin_minimap.matcher;
	matcher->detect_and_compute(Resources::getInstance().MapTemplate, keypoints, descriptors);

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