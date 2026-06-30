#pragma once
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

class GIMapDownloaderImpl;

class GIMapDownloader {
public:
	static GIMapDownloader& getInstance();
	/**
	 * @brief 设置本地依赖列表文件路径
	 * @param path 本地依赖列表文件路径
	 * @return 成功返回true，失败返回false
	 */
    bool setDependentsJsonPath(const fs::path& path);

	/**
	 * @brief 设置下载服务器地址
	 * @param host 下载服务器地址
	 * @return 成功返回true，失败返回false
	 */
	bool setHost(const std::string& host);

	/**
	 * @brief 设置下载目标路径
	 * @param path 下载目标路径
	 * @return 成功返回true，失败返回false
	 */
	bool setLocalPath(const fs::path& path);

	/**
	 * @brief 执行下载操作
	 * @return 是否成功
	 */
	bool download();

private:
	GIMapDownloader();
	GIMapDownloader(const GIMapDownloader&) = delete;
	GIMapDownloader& operator=(const GIMapDownloader&) = delete;
	GIMapDownloader(GIMapDownloader&&) = delete;
	GIMapDownloader& operator=(GIMapDownloader&&) = delete;

	friend class GIMapDownloaderImpl;
	std::unique_ptr<GIMapDownloaderImpl> pImpl;
};