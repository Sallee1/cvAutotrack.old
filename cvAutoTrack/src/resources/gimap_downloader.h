#pragma once
#include <memory>

class GIMapDownloaderImpl;

class GIMapDownloader {
public:
	static GIMapDownloader& getInstance();
	/**
	 * @brief 设置本地依赖列表文件路径
	 * @param path 本地依赖列表文件路径
	 * @return 成功返回true，失败返回false
	 */
	bool setDependentsJsonPath(const std::string& path);

	/*
	* @ brief 检查本地依赖是否和服务器一致 
	* @return 依赖一致返回true，不一致返回false
	*/
	bool verifyDependents();
	
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
	bool setLocalPath(const std::string& path);

	/**
	 * @brief 执行下载操作
	 * @return 是否成功
	 */
	bool download();

	cv::Mat getGIMapImg() const;
	auto getLayerMapper() const -> std::map<std::string, std::pair<cv::Rect2i, cv::Rect2i>>;
	auto getMapMapper() const -> std::map<std::pair<std::string, int>, std::pair<cv::Rect2i, cv::Rect2i>>;
private:
	GIMapDownloader();
	GIMapDownloader(const GIMapDownloader&) = delete;
	GIMapDownloader& operator=(const GIMapDownloader&) = delete;
	GIMapDownloader(GIMapDownloader&&) = delete;
	GIMapDownloader& operator=(GIMapDownloader&&) = delete;

	friend class GIMapDownloaderImpl;
	std::unique_ptr<GIMapDownloaderImpl> pImpl;
};