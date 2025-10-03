#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <filesystem>
#include "version/Version.h"
#include "serialize.h"

class MapKeypointCache {
public:
	std::string bulid_time;
	std::string bulid_version;
	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;
	std::string bulid_version_end;

	MapKeypointCache() {}

	MapKeypointCache(std::string bulid_time,
		std::string bulid_version,
		std::vector<cv::KeyPoint> keyPoints,
		cv::Mat descriptors) :
		bulid_time(bulid_time), bulid_version(bulid_version),
		keyPoints(keyPoints), descriptors(descriptors), bulid_version_end(bulid_version) {
	}

	void serialize(std::string outfileName);
	void deSerialize(std::string infileName, bool version_only = false);
};

struct GenshinMinimap; // 前向声明

bool save_map_keypoint_cache(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
bool load_map_keypoint_cache(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
bool get_map_keypoint(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
