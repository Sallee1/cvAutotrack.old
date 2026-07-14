#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;
#include "version/Version.h"
#include "serialize.h"

struct MapKeypointCache {
public:
	std::string bulid_time;            // metadata.json 的 update_time
	std::string bulid_version;         // DLL版本号 + layer_version（用#分隔）
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	std::string bulid_version_end;

	// 用于快速查找keypoints的LSH/grid元数据
	cv::Size2i map_size{};               // 地图的总大小
	cv::Point2i map_origin{};            // 地图左上角原点坐标（保存 bounds 偏移）
	cv::Size2i lsh_cell{};               // lsh的分块大小
	cv::Size2i lsh_grid_dims{};          // lsh的网格尺寸
	std::vector<int> lsh_cell_offsets;   // 分块的前缀和 size = cells+1
	std::vector<int> lsh_kp_indices;     // 线性展开的keypoint索引

	// 用于分块生成keypoints的元数据
	std::vector<cv::Rect2i> block_rects; // 分块的边界（不包含padding）
	std::vector<int> block_offsets;      // 分块的前缀和 size = blocks+1

	MapKeypointCache() {}

	MapKeypointCache(std::string bulid_time,
		std::string bulid_version,
		std::vector<cv::KeyPoint> keypoints,
		cv::Mat descriptors) :
		bulid_time(bulid_time), bulid_version(bulid_version),
		keypoints(keypoints), descriptors(descriptors), bulid_version_end(bulid_version) {
	}

	bool serialize(const fs::path& outfileName);
	bool deSerialize(const fs::path& infileName, bool version_only = false);
};

// 用于bbox查询的简单基于网格的LSH
struct KeypointGridLSH {
	cv::Rect2i bounds{};			//网格边界
	cv::Size2i cell{ 256, 256 };	//分块大小
	cv::Size2i dims{};				//网格尺寸
	std::vector<int> cell_offsets;	//分块的前缀和 size = cells+1
	std::vector<int> kp_indices;	//线性展开的keypoint索引
	std::vector<cv::Point2f> kp_pos; //关键点坐标

	/**
	 * @brief 根据给定的关键点、边界和单元格大小进行构建操作。
	 * @param kps 关键点的向量列表。
	 * @param bounds_ 表示区域边界的矩形。
	 * @param cell_ 单元格的尺寸。
	 */
	void build(const std::vector<cv::KeyPoint>& kps, const cv::Rect2i& bounds_, const cv::Size2i& cell_);

	/**
	 * @brief 从缓存对象中加载关键点数据。
	 * @param cache 关键点缓存对象，包含要加载的数据。
	 */
	void fromCache(const MapKeypointCache& cache);
	// return indices of keypoints contained in bbox

	/**
	 * @brief 查询给定边界框内的关键点索引。
	 * @param bbox 表示查询区域的矩形边界框。
	 * @return 返回包含在边界框内的关键点索引的向量。
	 */
	std::vector<int> query(const cv::Rect2i& bbox) const;

	/**
	 * @brief 查询给定边界框内的关键点索引（别名函数）。
	 * @param bbox 表示查询区域的矩形边界框。
	 * @return 返回包含在边界框内的关键点索引的向量。
	 */
	std::vector<int> query_indices(const cv::Rect2i& bbox) const { return query(bbox); }

	/**
	 * @brief 从关键点索引重建关键点和描述符。
	 * @param cache 输入缓存文件
	 * @param indices 关键点索引
	 * @param out_kps 输出关键点
	 * @param out_desc 输出描述子
	 */
	void gather_by_indices(
		const std::vector<int>& indices, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
		std::vector<cv::KeyPoint>& out_kps,
		cv::Mat& out_desc) const;

	/**
	 * @brief 在指定的边界框内直接查询关键点，并收集其描述符。
	 * @param bbox 用于查询的二维矩形边界框。
	 * @param cache 关键点缓存，用于检索关键点信息。
	 * @param out_kps 输出参数，用于存储查询到的关键点。
	 * @param out_desc 输出参数，用于存储关键点的描述符矩阵。
	 */
	void query_and_gather(const cv::Rect2i& bbox, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
		std::vector<cv::KeyPoint>& out_kps,
		cv::Mat& out_desc) const;
};

class IMatcher; // 前向声明

/**
 * @brief 生成/加载特征点缓存
 * @param matcher 特征匹配器（用于 detect+compute）
 * @return 缓存对象
 */
MapKeypointCache get_map_keypoint(const std::shared_ptr<IMatcher>& matcher);
