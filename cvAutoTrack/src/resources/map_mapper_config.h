#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#ifdef _CVAT_DEBUG
#include <resources/Resources.h>
#endif


namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace TianLi
{
	/**
	 * @brief 单个 MAP 的坐标映射参数
	 * 变换公式: output_x = input_x * scale_x + offset_x
	 *           output_y = input_y * scale_y + offset_y
	 */
	struct MapMapperEntry
	{
		std::string prefix;   // 如 "MAP:Teyvat"
		int map_id = 0;       // 0=Teyvat, 1=渊下宫, 2=地下层岩, 3=旧日之海, 4=远古圣山, 5=空之神殿
		double center_x = 0.0;
		double center_y = 0.0;
		double offset_x = 0.0;
		double offset_y = 0.0;
		double scale_x = 1.0;
		double scale_y = 1.0;

		/** 应用变换 */
		void apply(double& x, double& y) const
		{
			x = (x + center_x) * scale_x;
            y = (y + center_y) * scale_y;
		}

	};

	/**
	 * @brief 瓦片图像信息（从 metadata.json 展开）
	 */
	struct TileImageInfo
	{
		std::string file_path;      // 相对于资源目录的路径 (assets/xxx.avif)
		std::string md5;            // 文件 MD5
		std::string map_prefix;     // 所属 MAP 前缀 (如 "MAP:Teyvat")
		int map_id = 0;             // map_id
		int zoom = 1;               // 缩放级别
		double rect_x, rect_y;      // 在原始坐标空间中的位置
		double rect_w, rect_h;      // 覆盖范围
	};

	/**
	 * @brief MAP 边界信息（用于运行时命中检测）
	 */
	struct MapBoundInfo
	{
		int map_id = 0;
		std::string prefix;
		cv::Rect2d bounds;  // 在输出坐标空间中的边界
	};

	/**
	 * @brief 地图映射管理器
	 * 单例，负责加载 metadata.json，管理 MAP 映射参数，提供坐标转换服务。
	 */
	class MapMapperManager
	{
	public:
		static MapMapperManager& getInstance()
		{
			static MapMapperManager instance;
			return instance;
		}

		/** 从 metadata.json 文件加载 */
		bool loadFromFile(const fs::path& jsonPath)
		{
			std::ifstream ifs(jsonPath);
			if (!ifs.is_open()) return false;

			Json root;
			try { ifs >> root; }
			catch (...) { return false; }

			resourceDir = fs::path(jsonPath).parent_path().u8string();
			return parseFromJson(root);
		}

		/** 从包含 metadata.json 的目录加载 */
		bool loadFromDir(const fs::path& dirPath)
		{
			fs::path metaPath = dirPath / "metadata.json";
			return loadFromFile(metaPath);
		}

		/** 是否已加载 */
		bool isLoaded() const { return loaded; }

		// ============ 查询接口 ============

		/**
		 * @brief 通过 fullname 前缀匹配查找 MAP 映射
		 * @return 找到返回 entry 指针，否则 nullptr
		 */
		const std::optional<MapMapperEntry> findMapper(const std::string& fullname) const
		{
			// 最长前缀优先匹配
			std::optional<MapMapperEntry> best = std::nullopt;
			size_t bestLen = 0;
			for (const auto& [id, entry] : mappers)
			{
                std::string prefix = entry.prefix;
				if (fullname.compare(0, prefix.size(), prefix) == 0 && prefix.size() > bestLen)
				{
					best = entry;
					bestLen = prefix.size();
				}
			}
			return best;
		}

		/**
		 * @brief 对坐标应用指定 fullname 对应的 MAP 变换
		 */
		bool transform(const std::string& fullname, double& x, double& y) const
		{
			auto entry = findMapper(fullname);
			if (!entry) return false;
			entry->apply(x, y);
			return true;
		}

		/**
		 * @brief 获取 fullname 对应的 map_id
		 */
		int getMapId(const std::string& fullname) const
		{
			auto entry = findMapper(fullname);
			return entry ? entry->map_id : 0;
		}

		/**
		 * @brief 查找坐标所在特殊地图（运行时）
		 * @param x,y 输入坐标（在统一输出空间中）
		 * @return (坐标, map_id)，map_id=0 表示未匹配特殊地图（即 Teyvat）
		 */
		std::pair<cv::Point2d, int> convertPosition(double x, double y) const
		{
            int out_id = 0;
            cv::Point2d out_pos{ x,y };
            //TODO: 待重新实现地区检测
			for (const auto& [id,bi] : bounds)
			{
                //跳过概率最大的提瓦特
                if (id == 0) continue;
				if (bi.bounds.contains(cv::Point2d(x, y)))
				{
                    out_pos.x -= mappers.at(id).offset_x;
                    out_pos.y -= mappers.at(id).offset_y;
                    out_id = id;
                    break;
				}
			}
#ifdef _CVAT_DEBUG
            //去除为了绘图添加的额外偏移量
            out_pos.x -= Resources::getInstance().DebugParams.offset.x;
            out_pos.y -= Resources::getInstance().DebugParams.offset.y;
#endif
            
            //根据id映射apply应用坐标
            mappers.at(out_id).apply(out_pos.x, out_pos.y);

			return { out_pos,out_id };
		}

		// ============ 版本信息 ============

		/**
		 * @brief 获取 metadata 的 layer_version（如 "1.1"）
		 */
		const std::string& getLayerVersion() const { return layerVersion; }

		
		/**
		 * @brief 获取 metadata 的 game_version（如 "1.0.0"）
		 */
		const std::string& getGameVersion() const { return gameVersion; }


		/**
		 * @brief 获取 metadata 的 update_time
		 */
		const std::string& getUpdateTime() const { return updateTime; }

		/**
		 * @brief 获取 layer_version 的大版本号（如 "1.1" → 1）
		 */
		int getMajorLayerVersion() const
		{
			if (layerVersion.empty()) return 1;
			auto pos = layerVersion.find('.');
			try {
				return std::stoi(layerVersion.substr(0, pos));
			}
			catch (...) { return 1; }
		}

		/** 支持的大版本号（硬编码，与破坏性更新同步） */
		static constexpr int SUPPORTED_MAJOR_VERSION = 1;

		/**
		 * @brief 检查大版本是否兼容
		 */
		bool isVersionCompatible() const
		{
			return getMajorLayerVersion() == SUPPORTED_MAJOR_VERSION;
		}

		// ============ 数据访问 ============

		const std::vector<TileImageInfo>& getTileInfos() const { return tileInfos; }
		const std::string& getResourceDir() const { return resourceDir; }
		const auto& getMappers() const { return mappers; }
        const auto& getBounds() const { return bounds; }

	private:
		MapMapperManager() = default;

		bool parseFromJson(const Json& root)
		{
			try
			{
				mappers.clear();
				tileInfos.clear();
				bounds.clear();

				// 解析版本信息
				layerVersion = root.value("layer_version", "1.0");
				gameVersion = root.value("game_version", "1.0.0");
				updateTime = root.value("update_time", "");

				if (!isVersionCompatible())
				{
					return false;
				}

				// 解析 map_mapper
				if (root.contains("map_mapper") && root["map_mapper"].is_object())
				{
					for (auto& [key, val] : root["map_mapper"].items())
					{
						MapMapperEntry entry;
						entry.map_id = val.value("map_id", 0);
                        entry.prefix = key;

						
						if (val.contains("offset") && val["offset"].is_array() && val["offset"].size() >= 2)
						{
							entry.offset_x = val["offset"][0].get<double>();
							entry.offset_y = val["offset"][1].get<double>();
						}

						if (val.contains("center") && val["center"].is_array() && val["center"].size() >= 2)
						{
							entry.center_x = val["center"][0].get<double>();
							entry.center_y = val["center"][1].get<double>();
						}
						if (val.contains("scale") && val["scale"].is_array() && val["scale"].size() >= 2)
						{
							entry.scale_x = val["scale"][0].get<double>();
							entry.scale_y = val["scale"][1].get<double>();
						}

						mappers[entry.map_id] = entry;
					}
				}

				// 解析 layers，展开 tile
				if (root.contains("layers") && root["layers"].is_array())
				{
					for (auto& layerJson : root["layers"])
					{
						std::string fullname = layerJson.value("fullname", "");
						std::string name = layerJson.value("name", "");

						// 计算 map_prefix
						auto mapper = findMapper(fullname);
						std::string mapPrefix = mapper ? mapper->prefix : "MAP:Teyvat";
						int mapId = mapper ? mapper->map_id : 0;

						if (layerJson.contains("images") && layerJson["images"].is_array())
						{
							for (auto& imgJson : layerJson["images"])
							{
								TileImageInfo info;
								info.file_path = imgJson.value("path", "");
								info.md5 = imgJson.value("md5", "");
								info.zoom = imgJson.value("zoom", 1);
								info.map_prefix = mapPrefix;
								info.map_id = mapId;

								if (imgJson.contains("rect") && imgJson["rect"].is_array() && imgJson["rect"].size() >= 4)
								{
									info.rect_x = imgJson["rect"][0].get<double>();
									info.rect_y = imgJson["rect"][1].get<double>();
									info.rect_w = imgJson["rect"][2].get<double>();
									info.rect_h = imgJson["rect"][3].get<double>();

#ifdef _CVAT_DEBUG
                                    //如果是调试模式，加上偏移量用于叠图
                                    info.rect_x += Resources::getInstance().DebugParams.offset.x;
                                    info.rect_y += Resources::getInstance().DebugParams.offset.y;
#endif
								}

								tileInfos.push_back(info);
							}
						}
					}
				}

				// 计算各 MAP 的边界
				computeBounds();

				loaded = true;
				return true;
			}
			catch (...)
			{
				return false;
			}
		}

		//TODO: 这里假定不同地区坐标不重叠
		//如果坐标重叠搞不了
		void computeBounds()
		{
			// 按 map_id 分组计算边界
			std::map<int, cv::Rect2d> mapBounds;
			std::map<int, std::string> mapPrefix;

			for (auto& [id, entry] : mappers)
			{
				mapBounds[entry.map_id] = cv::Rect2d();
				mapPrefix[entry.map_id] = entry.prefix;
			}

			for (auto& tile : tileInfos)
			{
				auto it = mapBounds.find(tile.map_id);
				if (it == mapBounds.end()) continue;

				// 将 tile rect 的角点转换到输出空间
                cv::Point2d tile_lt = { tile.rect_x,tile.rect_y };
                cv::Point2d tile_rb = { tile.rect_x + tile.rect_w, tile.rect_y + tile.rect_h};

				// 应用 map_mapper 变换
				auto& entry = mappers.at(tile.map_id);

				cv::Rect2d& b = it->second;
				if (b.width == 0 && b.height == 0)
				{
					b = cv::Rect2d(tile_lt, tile_rb);
				}
				else
				{
					b |= cv::Rect2d(tile_lt, tile_rb);
				}
			}

			// 写入 bounds 列表
			bounds.clear();
			for (auto& [id, rect] : mapBounds)
			{
                cv::Rect2d rect_with_offset = rect;
                rect_with_offset.x += mappers.at(id).offset_x;
                rect_with_offset.y += mappers.at(id).offset_y;
				bounds[id] = MapBoundInfo{ id, mapPrefix[id], rect_with_offset };
			}
		}

	private:
		std::map<int, MapMapperEntry> mappers;      // 重映射信息
		std::map<int, MapBoundInfo> bounds;         // 通过tile计算的边界信息
		std::vector<TileImageInfo> tileInfos;       // tile列表
		std::string resourceDir;
		std::string layerVersion;
		std::string gameVersion;
		std::string updateTime;
		bool loaded = false;
	};
}
