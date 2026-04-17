#include <pch.h>
#include "KeypointsCache.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <execution>
#include "version/Version.h"
#include "serialize.h"
#include "resources/Resources.h"
#include "resources/map_mapper.h"
#include "utils/utils.progress.h"

void MapKeypointCache::serialize(std::string outFileName)
{
	std::ofstream ofs(outFileName, std::fstream::out | std::fstream::binary);
	Tianli::Resources::Utils::serializeStream ss(ofs);
	ss << this->bulid_time;
	ss << this->bulid_version;
	ss << this->keypoints;
	ss << this->descriptors;

	// LSH元数据相关信息（兼容旧版结构）
	ss << this->map_size;
	ss << this->lsh_cell;
	ss << this->lsh_grid_dims;
	ss << this->lsh_cell_offsets;
	ss << this->lsh_kp_indices;
	ss << this->block_rects;
	ss << this->block_offsets;

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

	dss >> this->keypoints;
	dss >> this->descriptors;

	//读取LSH元数据相关信息（可选）
	try {
		// 成功读取
		dss >> this->map_size;
		dss >> this->lsh_cell;
		dss >> this->lsh_grid_dims;
		dss >> this->lsh_cell_offsets;
		dss >> this->lsh_kp_indices;
		dss >> this->block_rects;
		dss >> this->block_offsets;
	}
	catch (...) {
		// 读取失败，则可能是旧版本缓存
		this->map_size = {};
		this->lsh_cell = {};
		this->lsh_grid_dims = {};
		this->lsh_cell_offsets.clear();
		this->lsh_kp_indices.clear();
		this->block_rects.clear();
		this->block_offsets.clear();
	}
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

	bool gen_map_keypoint_cache(const GenshinMinimap& genshin_minimap, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
		std::vector<cv::Rect2i>& block_rects, std::vector<int>& block_offsets)
	{
		auto& matcher = genshin_minimap.matcher;
		auto& map_template = Resources::getInstance().MapTemplate;

		//分块处理
		int block_size = 1200;
		int padding = 64; //64像素重叠
		auto rects = getRects(cv::Rect2i(0, 0, map_template.cols, map_template.rows), cv::Size2i(block_size, block_size), cv::Size2i(padding, padding));

		block_rects.clear();
		block_offsets.clear();
		block_offsets.push_back(0);
		TianLi::Utils::Win32ProgressWindow progress_window;
		progress_window.create(L"cvAutoTrack", static_cast<int>(rects.size()), L"正在生成特征点缓存...");

		for (size_t i = 0; i < rects.size(); ++i)
		{
			auto& rect = rects[i];
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
			{
				block_rects.push_back(inner_rect);
				block_offsets.push_back(block_offsets.back());
				progress_window.set_status(L"正在生成特征点缓存: " + std::to_wstring(i + 1) + L"/" + std::to_wstring(rects.size()));
				progress_window.set_value(static_cast<int>(i + 1));
				continue;
			}

			//计算描述子
			matcher->compute(map_template(rect.first), kps, desc);

			//调整关键点坐标到全图坐标系
			for (auto& kp : kps)
			{
				kp.pt.x += rect.first.x;
				kp.pt.y += rect.first.y;
			}
			//合并关键点和描述子，确保同一分块的描述子是连续追加
			if (keypoints.empty())
			{
				keypoints = kps;
				descriptors = desc;
			}
			else
			{
				cv::vconcat(descriptors, desc, descriptors);
				keypoints.insert(keypoints.end(), kps.begin(), kps.end());
			}

			block_rects.push_back(inner_rect);
			block_offsets.push_back(block_offsets.back() + static_cast<int>(kps.size()));
			progress_window.set_status(L"正在生成特征点缓存: " + std::to_wstring(i + 1) + L"/" + std::to_wstring(rects.size()));
			progress_window.set_value(static_cast<int>(i + 1));
		}
		progress_window.close();
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

// Build a simple grid based LSH on keypoints
void KeypointGridLSH::build(const std::vector<cv::KeyPoint>& kps, const cv::Rect2i& bounds_, const cv::Size2i& cell_)
{
	bounds = bounds_;
	cell = cell_;

	int gw = std::max(1, (bounds.width + cell.width - 1) / cell.width);
	int gh = std::max(1, (bounds.height + cell.height - 1) / cell.height);
	dims = { gw, gh };
	int cells = gw * gh;
	std::vector<int> counts(cells, 0);

	auto cell_index = [&](const cv::Point2f& pt) {
		int cx = std::clamp((int)((pt.x - bounds.x) / cell.width), 0, gw - 1);
		int cy = std::clamp((int)((pt.y - bounds.y) / cell.height), 0, gh - 1);
		return cy * gw + cx;
		};

	for (const auto& kp : kps)
	{
		counts[cell_index(kp.pt)]++;
	}

	cell_offsets.assign(cells + 1, 0);
	for (int i = 0; i < cells; ++i) cell_offsets[i + 1] = cell_offsets[i] + counts[i];
	kp_indices.assign(kps.size(), -1);

	std::vector<int> cursor = cell_offsets;
	for (int i = 0; i < (int)kps.size(); ++i)
	{
		int c = cell_index(kps[i].pt);
		kp_indices[cursor[c]++] = i;
	}

	kp_pos.assign(kps.size(), { 0,0 });
	cv::parallel_for_({ 0, static_cast<int>(kps.size()) }, [&](const cv::Range& r)
		{
			for (int i = r.start; i < r.end; ++i)
			{
				kp_pos[i] = kps[i].pt;
			}
		});
}

void KeypointGridLSH::fromCache(const MapKeypointCache& cache)
{
	bounds = { 0, 0, cache.map_size.width, cache.map_size.height };
	cell = cache.lsh_cell;
	dims = cache.lsh_grid_dims;
	cell_offsets = cache.lsh_cell_offsets;
	kp_indices = cache.lsh_kp_indices;

	auto& kps = cache.keypoints;
	kp_pos.assign(kps.size(), { 0,0 });
	cv::parallel_for_({ 0, static_cast<int>(kps.size()) }, [&](const cv::Range& r)
		{
			for (int i = r.start; i < r.end; ++i)
			{
				kp_pos[i] = kps[i].pt;
			}
		});
}

std::vector<int> KeypointGridLSH::query(const cv::Rect2i& bbox) const
{
	std::vector<int> result;
	if (kp_pos.empty() && kp_indices.empty()) return result;
	int gw = dims.width, gh = dims.height;
	if (gw == 0 || gh == 0) return result;
	cv::Rect2i clip = bbox;

	if (clip.width <= 0 || clip.height <= 0) return result;
	int x0 = std::clamp((clip.x - bounds.x) / cell.width, 0, gw - 1);
	int y0 = std::clamp((clip.y - bounds.y) / cell.height, 0, gh - 1);
	int x1 = std::clamp((clip.x + clip.width - 1 - bounds.x) / cell.width, 0, gw - 1);
	int y1 = std::clamp((clip.y + clip.height - 1 - bounds.y) / cell.height, 0, gh - 1);

	for (int cy = y0; cy <= y1; ++cy)
	{
		for (int cx = x0; cx <= x1; ++cx)
		{
			size_t c = static_cast<size_t>(cy * gw + cx);
			int begin = cell_offsets[c];
			int end = cell_offsets[c + 1];
			for (int i = begin; i < end; ++i)
			{
				int idx = kp_indices[i];
				const auto& p = kp_pos[idx];
				if (clip.contains(p))
					result.push_back(idx);
			}
		}
	}
	return result;
}

void KeypointGridLSH::gather_by_indices(
	const std::vector<int>& indices, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
	std::vector<cv::KeyPoint>& out_kps,
	cv::Mat& out_desc) const
{
	out_kps.resize(indices.size());
	if (indices.empty()) { out_desc.release(); return; }

	// clone descriptors header (type, cols)
	int desc_cols = descriptors.cols;
	int desc_type = descriptors.type();
	out_desc.create((int)indices.size(), desc_cols, desc_type);

	// parallel copy rows by indices
	cv::parallel_for_({ 0, (int)indices.size() }, [&](const cv::Range& r)
		{
			for (int i = r.start; i < r.end; ++i)
			{
				int idx = indices[i];
				out_kps[i] = keypoints[idx];
				descriptors.row(idx).copyTo(out_desc.row(i));
			}
		});
}

void KeypointGridLSH::query_and_gather(const cv::Rect2i& bbox,
	const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
	std::vector<cv::KeyPoint>& out_kps,
	cv::Mat& out_desc) const
{
	auto idx = query(bbox);
	gather_by_indices(idx, keypoints, descriptors, out_kps, out_desc);
}

bool save_map_keypoint_cache(const GenshinMinimap& genshin_minimap, MapKeypointCache& cache)
{
	std::vector<cv::Rect2i> block_rects;
	std::vector<int> block_offsets;
	gen_map_keypoint_cache(genshin_minimap, cache.keypoints, cache.descriptors, block_rects, block_offsets);
	remap_keypoints(cache.keypoints);
	std::string build_time = __DATE__ " " __TIME__;
	cache.bulid_time = build_time;
	cache.bulid_version = TianLi::Version::build_version;

	// fill LSH/grid metadata
	auto cell_size = Resources::getInstance().lsh_cell_size;
	cache.map_size = { Resources::getInstance().MapTemplate.cols, Resources::getInstance().MapTemplate.rows };
	cache.lsh_cell = { cell_size, cell_size };
	int gw = std::max(1, (cache.map_size.width + cache.lsh_cell.width - 1) / cache.lsh_cell.width);
	int gh = std::max(1, (cache.map_size.height + cache.lsh_cell.height - 1) / cache.lsh_cell.height);
	cache.lsh_grid_dims = { gw, gh };

	// Build grid on remapped keypoints
	KeypointGridLSH grid;
	grid.build(cache.keypoints, { 0,0, cache.map_size.width, cache.map_size.height }, cache.lsh_cell);
	cache.lsh_cell_offsets = grid.cell_offsets;
	cache.lsh_kp_indices = grid.kp_indices;

	// record block grouping
	cache.block_rects = block_rects;
	cache.block_offsets = block_offsets;

	std::filesystem::remove("cvAutoTrack_Cache.xml");
	cache.serialize("cvAutoTrack_Cache.xml");

	return true;
}

bool load_map_keypoint_cache(MapKeypointCache& cache)
{
	if (std::filesystem::exists("cvAutoTrack_Cache.xml") == false)
	{
		return false;
	}

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

	return true;
}

MapKeypointCache get_map_keypoint(const GenshinMinimap& genshin_minimap)
{
	MapKeypointCache cache;
	if (load_map_keypoint_cache(cache) == false)
	{
		cache = {};
		save_map_keypoint_cache(genshin_minimap, cache);
	}
	return cache;
}
