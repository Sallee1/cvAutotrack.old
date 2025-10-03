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