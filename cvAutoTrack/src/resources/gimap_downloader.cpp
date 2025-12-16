#include "pch.h"
#include "gimap_downloader.h"
#include "downloader/cfiledownloaderasync.h"
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using Json = nlohmann::json;

class GIMapDownloaderImpl {
public:
	GIMapDownloaderImpl() = default;
	~GIMapDownloaderImpl() = default;
	tianli::FileDownloaderAsync downloader{ 32 };
	std::string host;
	fs::path local_path;
	fs::path dependents_json_path;
	Json local_dependents_json;
	Json remote_dependents_json;
};

GIMapDownloader& GIMapDownloader::getInstance() {
	static GIMapDownloader instance;
	return instance;
}

bool GIMapDownloader::setDependentsJsonPath(const std::string& path)
{
	if (fs::exists(path))
	{
		pImpl->dependents_json_path = path;
		return true;
	}
	return false;
}

bool GIMapDownloader::verifyDependents()
{
	return false;
}

bool GIMapDownloader::setHost(const std::string& host)
{
	// 下载dependents.json文件做测试
	std::string new_dependents_url = host + "/dependents.json";
	std::string new_dependents_path = fs::temp_directory_path().append("dependents.json").string();
	std::cout << "Downloading dependents.json from: " << new_dependents_url << " to " << new_dependents_path << std::endl;
	pImpl->downloader.addTask(new_dependents_path, new_dependents_url);
	for (int i = 0; i < 3; i++)
	{
		pImpl->downloader.wait();
		if (pImpl->downloader.getFailed().empty())
		{
			std::ifstream ifs(new_dependents_path);
			if (ifs.is_open())
			{
				try {
					ifs >> pImpl->remote_dependents_json;
					pImpl->host = host;
					return true;
				}
				catch (const std::exception& e) {
					std::cerr << "Error parsing remote dependents.json: " << e.what() << std::endl;
				}
			}
		}
	}

	return false;
}

bool GIMapDownloader::download()
{
	return false;
}

cv::Mat GIMapDownloader::getGIMapImg() const
{
	return cv::Mat();
}

auto GIMapDownloader::getLayerMapper() const -> std::map<std::string, std::pair<cv::Rect2i, cv::Rect2i>>
{
	return std::map<std::string, std::pair<cv::Rect2i, cv::Rect2i>>();
}

auto GIMapDownloader::getMapMapper() const -> std::map<std::pair<std::string, int>, std::pair<cv::Rect2i, cv::Rect2i>>
{
	return std::map<std::pair<std::string, int>, std::pair<cv::Rect2i, cv::Rect2i>>();
}

GIMapDownloader::GIMapDownloader() :pImpl(std::make_unique<GIMapDownloaderImpl>())
{
}