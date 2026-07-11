#include <pch.h>
#include "KeypointsCache.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <execution>
#include <mutex>
#include <atomic>
#include "version/Version.h"
#include "serialize.h"
#include "resources/Resources.h"
#include "resources/map_mapper_config.h"
#include "utils/utils.progress.h"
#include "match/IMatcher.h"

bool MapKeypointCache::serialize(std::string outFileName)
{
	std::ofstream ofs(outFileName, std::fstream::out | std::fstream::binary);
	TianLi::Utils::serializeStream ss(ofs);
	ss << this->bulid_time;
	ss << this->bulid_version;
	ss << this->keypoints;
	ss << this->descriptors;

	// LSH元数据相关信息（兼容旧版结构）
	ss << this->map_size;
	ss << this->map_origin;
	ss << this->lsh_cell;
	ss << this->lsh_grid_dims;
	ss << this->lsh_cell_offsets;
	ss << this->lsh_kp_indices;
	ss << this->block_rects;
	ss << this->block_offsets;

	ss << this->bulid_version_end;
	ss.align();
	ofs.close();

    return true;
}

bool MapKeypointCache::deSerialize(std::string infileName, bool version_only)
{
    try {
	    std::ifstream ifs(infileName, std::fstream::out | std::fstream::binary);
        if (!ifs.is_open())
        {
            return false;
        }

	    TianLi::Utils::deSerializeStream dss(ifs);
        dss >> this->bulid_time;
        dss >> this->bulid_version;

        if (version_only)
        {
            ifs.close();
            return true;
        }

        dss >> this->keypoints;
        dss >> this->descriptors;

	    //读取LSH元数据相关信息（可选）
		dss >> this->map_size;
		dss >> this->map_origin;
		dss >> this->lsh_cell;
		dss >> this->lsh_grid_dims;
		dss >> this->lsh_cell_offsets;
		dss >> this->lsh_kp_indices;
		dss >> this->block_rects;
		dss >> this->block_offsets;

	    dss >> this->bulid_version_end;
	    ifs.close();
        return true;
    }
    catch (...)
    {
        //缓存读取失败，返回false
        return false;
    }
    return false;
}

namespace {
	bool gen_map_keypoint_cache(const std::shared_ptr<IMatcher>& matcher, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::vector<cv::Rect2i>& block_rects, std::vector<int>& block_offsets)
	{
		auto& mapper = TianLi::MapMapperManager::getInstance();
		const auto& tiles = mapper.getTileInfos();

		if (tiles.empty()) return false;

		keypoints.clear();
		descriptors = cv::Mat();

		TianLi::Utils::Win32ProgressWindow progress_window;
		progress_window.create(L"cvAutoTrack", static_cast<int>(tiles.size()), L"正在从瓦片生成特征点缓存...");

		// 每个瓦片的线程局部结果
		struct TileResult {
			std::vector<cv::KeyPoint> kps;
			cv::Mat desc;
			cv::Rect2i output_rect;  // 瓦片在输出坐标空间的边界
		};
		std::vector<TileResult> tileResults(tiles.size());
		std::atomic<bool> hasError{ false };
		std::atomic<int> progressCount{ 0 };
		std::mutex progressMutex;
		std::string first_missing_file;
		std::mutex missing_file_mutex;

		// 并行处理所有瓦片
		cv::parallel_for_({ 0, static_cast<int>(tiles.size()) },
			[&](const cv::Range& range)
			{
				for (int i = range.start; i < range.end; i++)
				{
					if (hasError) break;

					const auto& tile = tiles[i];

					// 加载瓦片图像
					fs::path imgPath = fs::u8path(mapper.getResourceDir()) / fs::u8path(tile.file_path);
					// imread依赖ACP编码路径
					if (!fs::exists(imgPath))
					{
						{
							std::lock_guard<std::mutex> lock(missing_file_mutex);
							if (first_missing_file.empty()) first_missing_file = tile.file_path;
						}
						hasError = true;
						return;
					}
					cv::Mat tileImg = cv::imread(imgPath.string(),cv::IMREAD_GRAYSCALE);

					if (tileImg.empty())
					{
						{
							std::lock_guard<std::mutex> lock(missing_file_mutex);
							if (first_missing_file.empty()) first_missing_file = tile.file_path;
						}
						hasError = true;
						return;
					}

					int img_w = tileImg.cols;
					int img_h = tileImg.rows;
					if (img_w <= 0 || img_h <= 0) continue;

					// 查找 MAP 变换
					auto& entry = mapper.getMappers().at(tile.map_id);

					// 检测关键点

                    //填充64px边距，确保边缘生成数据
                    int padding_size = 64;
                    cv::copyMakeBorder(tileImg, tileImg, padding_size, padding_size, padding_size, padding_size,cv::BORDER_CONSTANT);
					std::vector<cv::KeyPoint> kps;
					cv::Mat desc;
					matcher->detect(tileImg, kps);

					if (!kps.empty())
					{
                        // 清理界外点
                        kps.erase(std::remove_if(kps.begin(), kps.end(), [&](const cv::KeyPoint& kp) {
                            return !cv::Rect2i(padding_size, padding_size, img_w, img_h).contains(kp.pt);}), 
                        kps.end());

						// 计算描述子
						matcher->compute(tileImg, kps, desc);

						// 将像素坐标转换为输出坐标空间，并清理边界外点
						// 像素 → 原始坐标(tile rect) → MAP变换 → 输出坐标
						for (auto& kp : kps)
						{
                            kp.pt -= cv::Point2f{ static_cast<float>(padding_size) , static_cast<float>(padding_size) };      //去除填充的影响
							double raw_x = tile.rect_x + (kp.pt.x / img_w) * tile.rect_w;
							double raw_y = tile.rect_y + (kp.pt.y / img_h) * tile.rect_h;
							//计算附加偏移量
                            raw_x += entry.offset_x;
                            raw_y += entry.offset_y;
							kp.pt = cv::Point2f(static_cast<float>(raw_x), static_cast<float>(raw_y));
						}
					}

					// 计算瓦片在输出坐标空间的边界
					{
						double rx1 = tile.rect_x, ry1 = tile.rect_y;
						double rx2 = tile.rect_x + tile.rect_w, ry2 = tile.rect_y + tile.rect_h;

                        //计算附加偏移量
                        rx1 += entry.offset_x;
                        ry1 += entry.offset_y;
                        rx2 += entry.offset_x;
                        ry2 += entry.offset_y;


						int ox = static_cast<int>(std::floor(std::min(rx1, rx2)));
						int oy = static_cast<int>(std::floor(std::min(ry1, ry2)));
						int ow = static_cast<int>(std::ceil(std::abs(rx2 - rx1)));
						int oh = static_cast<int>(std::ceil(std::abs(ry2 - ry1)));
						tileResults[i].output_rect = cv::Rect2i(ox, oy, ow, oh);
					}

					tileResults[i].kps = std::move(kps);
					tileResults[i].desc = std::move(desc);

					// 更新进度（加锁保护 Win32 控件）
					int cur = ++progressCount;
					{
						std::lock_guard<std::mutex> lock(progressMutex);
						progress_window.set_value(cur);
						progress_window.set_status(L"正在从瓦片生成特征点缓存: " + std::to_wstring(cur) + L"/" + std::to_wstring(tiles.size()));
					}
				}
			}
		);

		progress_window.close();
		if (hasError)
		{
			// 在非并行主线程中弹窗警告
			std::string warn_msg = "特征点缓存生成失败：地图瓦片缺失！\n";
			if (!first_missing_file.empty())
				warn_msg += "缺少文件: " + first_missing_file + "\n";
			warn_msg += "请检查资源下载是否完整，将尝试使用旧缓存回退。";
			MessageBox(NULL, fs::u8path(warn_msg).wstring().c_str(), L"警告", MB_OK | MB_ICONWARNING);
			return false;
		}

		// 合并所有瓦片的结果 — 两遍扫描：先统计总量，再一次性预分配并逐块拷贝
		block_rects.clear();
		block_offsets.clear();
		block_offsets.push_back(0);

		// 第一遍：统计总关键点数和描述子参数
		int total_kps = 0;
		int desc_cols = 0;
		int desc_type = 0;
		for (auto& result : tileResults)
		{
			int n = static_cast<int>(result.kps.size());
			total_kps += n;
			if (n > 0 && desc_cols == 0)
			{
				desc_cols = result.desc.cols;
				desc_type = result.desc.type();
			}
		}

		// 预分配输出容器
		if (total_kps > 0)
		{
			descriptors.create(total_kps, desc_cols, desc_type);
			keypoints.reserve(total_kps);
		}

		// 第二遍：按块拷贝（利用 block_offsets 作为偏移量）
		int offset = 0;
		for (auto& result : tileResults)
		{
			int kp_count = static_cast<int>(result.kps.size());
			block_rects.push_back(result.output_rect);

			if (kp_count > 0)
			{
				result.desc.copyTo(descriptors.rowRange(offset, offset + kp_count));
				keypoints.insert(keypoints.end(),
					std::make_move_iterator(result.kps.begin()),
					std::make_move_iterator(result.kps.end()));
			}

			offset += kp_count;
			block_offsets.push_back(offset);
		}

		return true;
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
	bounds = { cache.map_origin.x, cache.map_origin.y, cache.map_size.width, cache.map_size.height };
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

/**
 * @brief 从关键点计算总地图边界，建 LSH 网格
 */
void build_lsh_grid(MapKeypointCache& cache)
{
	auto& mapper = TianLi::MapMapperManager::getInstance();
	auto cell_size = Resources::getInstance().lsh_cell_size;
	cache.lsh_cell = { cell_size, cell_size };

	// 计算输入边界时考虑 MAP 变换后的 tile 范围
	cv::Rect2d totalBounds;
	for (const auto& [id,bound] : mapper.getBounds())
	{
        if (totalBounds.width == 0 && totalBounds.height == 0)
        {
            totalBounds = bound.bounds;
        }
        else
        {
            totalBounds |= bound.bounds;
        }
	}

	// 统一用 Rect2i 作为 LSH 网格边界（和 build() 的隐式转换保持一致）
	cv::Rect2i totalBoundsInt = totalBounds;
	cache.map_origin = { totalBoundsInt.x, totalBoundsInt.y };

	KeypointGridLSH grid;
	grid.build(cache.keypoints, totalBoundsInt, cache.lsh_cell);

	cache.lsh_grid_dims = grid.dims;
	// map_size 存储 LSH 网格的实际覆盖范围 = 格子数 × 格子大小
	cache.map_size = {
		grid.dims.width * cache.lsh_cell.width,
		grid.dims.height * cache.lsh_cell.height
	};
	cache.lsh_cell_offsets = grid.cell_offsets;
	cache.lsh_kp_indices = grid.kp_indices;
}

bool save_map_keypoint_cache(const std::shared_ptr<IMatcher>& matcher, MapKeypointCache& cache)
{
	std::vector<cv::Rect2i> block_rects;
	std::vector<int> block_offsets;
	if (!gen_map_keypoint_cache(matcher, cache.keypoints, cache.descriptors, block_rects, block_offsets))
	{
		return false;
	}
	{
		auto& mapper = TianLi::MapMapperManager::getInstance();
		cache.bulid_version = TianLi::Version::build_version + "#L" + mapper.getLayerVersion() + "#G" + mapper.getGameVersion() + "@" + mapper.getUpdateTime();
		cache.bulid_version_end = cache.bulid_version;
		cache.bulid_time = __DATE__ " " __TIME__;		//不读，仅供dll兼容
	}



	// 从关键点计算总地图边界并建 LSH 网格
	if (!cache.keypoints.empty())
	{
		build_lsh_grid(cache);
	}

	// record block grouping
	cache.block_rects = block_rects;
	cache.block_offsets = block_offsets;

	std::filesystem::remove(Resources::getInstance().CachePath.cvAutoTrack_Cache);
    cache.serialize(Resources::getInstance().CachePath.cvAutoTrack_Cache);

	// 构建并缓存 FLANN 索引到独立文件
	std::error_code ec;
	fs::remove(Resources::getInstance().CachePath.cvAutoTrack_Cache_flann, ec);
    matcher->cache_flann_train_descriptors(cache.descriptors);
    matcher->save_flann_index(Resources::getInstance().CachePath.cvAutoTrack_Cache_flann);

    return true;
}

bool load_map_keypoint_cache(MapKeypointCache& cache)
{
    if (std::filesystem::exists(Resources::getInstance().CachePath.cvAutoTrack_Cache) == false)
    {
        return false;
    }

    // bulid_version 包含 "#" + layer_version，反序列化后直接比较即可
    if (cache.deSerialize(Resources::getInstance().CachePath.cvAutoTrack_Cache, true))
    {
        auto& mapper = TianLi::MapMapperManager::getInstance();
        if (cache.bulid_version != TianLi::Version::build_version + "#L" + mapper.getLayerVersion() + "#G" + mapper.getGameVersion() + "@" + mapper.getUpdateTime())
        {
            // 版本变更，缓存已过期；但保留文件作为回退，不删除
            return false;
        }
    }
    else
    {
        // 反序列化失败，缓存文件损坏，删除后重新生成
        std::error_code ec;
        fs::remove(Resources::getInstance().CachePath.cvAutoTrack_Cache, ec);
        fs::remove(Resources::getInstance().CachePath.cvAutoTrack_Cache_flann, ec);
        return false;
    }

    if (!cache.deSerialize(Resources::getInstance().CachePath.cvAutoTrack_Cache))
    {
        std::error_code ec;
        fs::remove(Resources::getInstance().CachePath.cvAutoTrack_Cache, ec);
        fs::remove(Resources::getInstance().CachePath.cvAutoTrack_Cache_flann, ec);
        return false;
    }

    if (cache.bulid_version != cache.bulid_version_end)    //写入不完整
    {
        // 保留文件作为回退
        return false;
    }

    // bulid_version 已包含 DLL版本 + layer_version + update_time
    // 反序列化后直接比较即可自动捕捉版本变更

    return true;
}

MapKeypointCache get_map_keypoint(const std::shared_ptr<IMatcher>& matcher)
{
    MapKeypointCache cache;
    auto& cache_path = Resources::getInstance().CachePath;

    // Stage 1: 尝试加载现有缓存（含版本校验）
    if (load_map_keypoint_cache(cache))
    {
        // 同步加载 FLANN 索引 — 优先从磁盘加载，失败则现场构建
        if (!matcher->try_load_flann_index(cache_path.cvAutoTrack_Cache_flann, cache.descriptors))
        {
            matcher->cache_flann_train_descriptors(cache.descriptors);
        }
        return cache;
    }

    // Stage 2: 缓存不存在或已过期，尝试重新生成
    cache = {};
    if (save_map_keypoint_cache(matcher, cache))
    {
        // save 内部已保存 FLANN 索引，无需重复操作
        return cache;
    }

    // Stage 3: 生成失败（如瓦片下载未完成），回退到旧缓存
    // 直接反序列化，不经过版本校验
    cache = {};
    cache.deSerialize(cache_path.cvAutoTrack_Cache);
    // 回退模式下只尝试从磁盘加载，不重建（避免耗时）
    if (!cache.keypoints.empty())
    {
        matcher->try_load_flann_index(cache_path.cvAutoTrack_Cache_flann, cache.descriptors);
    }
    return cache;
}
