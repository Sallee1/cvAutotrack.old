#include "pch.h"
#include "gimap_downloader.h"
#include "downloader/cfiledownloaderasync.h"
#include "downloader/cfiledownloader.h"
#include "utils/utils.progress.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <cctype>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using Json = nlohmann::json;

//=============================================================================
// 网络异常 —— 因网络问题导致的下载失败以异常形式抛出
//=============================================================================
class network_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

//=============================================================================
// 不区分大小写的字符串比较（用于 MD5 比对）
//=============================================================================
static bool iequals(const std::string& a, const std::string& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::toupper(static_cast<unsigned char>(a[i])) !=
            std::toupper(static_cast<unsigned char>(b[i])))
            return false;
    return true;
}

//=============================================================================
// PIMPL 实现
//=============================================================================
class GIMapDownloaderImpl {
public:
    GIMapDownloaderImpl() = default;
    ~GIMapDownloaderImpl() = default;

    //-------------------------------------------------------------------------
    // 校验：仅比对 sumMD5，忽略 JSON 格式差异
    //-------------------------------------------------------------------------
    bool verifyDependents()
    {
        if (local_dependents_json.is_null() || remote_dependents_json.is_null())
            return false;

        auto local_it = local_dependents_json.find("sumMD5");
        auto remote_it = remote_dependents_json.find("sumMD5");
        if (local_it == local_dependents_json.end() || remote_it == remote_dependents_json.end())
            return false;

        return iequals(local_it->get<std::string>(), remote_it->get<std::string>());
    }

    tianli::FileDownloaderAsync downloader{ 32 };
    std::string host;
    fs::path local_path;
    fs::path dependents_json_path;
    Json local_dependents_json;
    Json remote_dependents_json;
};

//=============================================================================
// GIMapDownloader 公开接口
//=============================================================================

GIMapDownloader& GIMapDownloader::getInstance()
{
    static GIMapDownloader instance;
    return instance;
}

//-------------------------------------------------------------------------
// setDependentsJsonPath —— 设置本地依赖列表路径，同时加载本地 dependents.json
//-------------------------------------------------------------------------
bool GIMapDownloader::setDependentsJsonPath(const fs::path& path)
{
    if (!fs::exists(path))
        return false;

    pImpl->dependents_json_path = fs::absolute(path);

    return true;
}

//-------------------------------------------------------------------------
// setLocalPath —— 设置下载目标路径，用于拼接相对路径
//-------------------------------------------------------------------------
bool GIMapDownloader::setLocalPath(const fs::path & path)
{
    if (path.empty())
        return false;
    pImpl->local_path = fs::absolute(path);
    return true;
}

//-------------------------------------------------------------------------
// setHost —— 连通性检查（HEAD 请求），不做实质性下载
//-------------------------------------------------------------------------
bool GIMapDownloader::setHost(const std::string& host)
{
    // 移除末尾多余的斜杠
    std::string base_host = host;
    while (!base_host.empty() && base_host.back() == '/')
        base_host.pop_back();

    tianli::FileDownloader::testConnection(base_host + "/dependents.json");

    pImpl->host = base_host;
    return true;
}

//-------------------------------------------------------------------------
// download —— 增量下载
//-------------------------------------------------------------------------
bool GIMapDownloader::download()
{
    // 1) 下载远程 dependents.json 到临时文件
    fs::path tmp_json = fs::temp_directory_path() / "gimap_remote_deps.json";
    {
        tianli::FileDownloader dl(tmp_json.string(), pImpl->host + "/dependents.json");
        if (!dl.download())
            throw network_error("下载远程 dependents.json 失败: [" +
                                std::to_string(dl.getLastErrorCode()) + "] " +
                                dl.getLastErrorMsg());
    }

    // 解析远程 JSON
    {
        std::ifstream ifs(tmp_json);
        if (!ifs.is_open())
            throw network_error("无法读取远程 dependents.json 临时文件");
        try {
            ifs >> pImpl->remote_dependents_json;
        }
        catch (const std::exception& e) {
            throw network_error("解析远程 dependents.json 失败: " + std::string(e.what()));
        }
    }
    std::error_code ec;
    fs::remove(tmp_json, ec);

    // 2) 加载本地 dependents.json（如果存在）
    {
        fs::path local_json = pImpl->dependents_json_path / "dependents.json";
        std::ifstream ifs(local_json);
        if (ifs.is_open())
        {
            try {
                ifs >> pImpl->local_dependents_json;
            }
            catch (const std::exception&) {
                // 本地 json 损坏，视为不存在，重新下载
            }
        }
    }

    // 3) 校验：sumMD5 一致则无需下载
    if (pImpl->verifyDependents())
        return true;

    // 2) 必须有远程列表
    if (pImpl->remote_dependents_json.is_null())
        throw network_error("缺少远程依赖列表，请先调用 setHost");

    auto filelist_it = pImpl->remote_dependents_json.find("filelist");
    if (filelist_it == pImpl->remote_dependents_json.end() || !filelist_it->is_array())
        throw network_error("远程 dependents.json 缺少 filelist 字段或格式错误");

    // 3) 确定下载根目录
    if (pImpl->local_path.empty())
        pImpl->local_path = pImpl->dependents_json_path;

    // 4) 增量筛查
    struct PendingFile {
        std::string filename;
        fs::path    target_path;   // 目标路径
        std::string url;
        std::string md5;
    };
    std::vector<PendingFile> pending;

    for (const auto& entry : *filelist_it)
    {
        std::string filename = entry["filename"].get<std::string>();
        std::string url     = entry["url"].get<std::string>();
        std::string md5     = entry.value("md5", "");

        fs::path target = pImpl->local_path / filename;
        bool need = true;

        if (fs::exists(target))
        {
            if (!md5.empty())
            {
                // 用 FileDownloader 的 MD5 校验能力
                tianli::FileDownloader checker(target.string(), "", md5);
                if (checker.download())   // 文件存在且 MD5 匹配 → 跳过
                    need = false;
            }
            else
            {
                // 无 MD5 但文件存在 → 跳过
                need = false;
            }
        }

        if (need)
        {
            fs::create_directories(fs::absolute(target).parent_path());
            pending.push_back({ filename, std::move(target), url, md5 });
        }
    }

    // 5) 全部已是最新：更新本地 dependents.json 后返回
    if (pending.empty())
    {
        pImpl->local_dependents_json = pImpl->remote_dependents_json;
        std::ofstream ofs(pImpl->dependents_json_path / "dependents.json");
        if (ofs.is_open()) ofs << pImpl->remote_dependents_json.dump(4);
        return true;
    }

    // 6) 创建进度窗口
    int total = static_cast<int>(pending.size());
    TianLi::Utils::Win32ProgressWindow progress;
    progress.create(L"更新地图资源", total, L"正在下载地图瓦片...");

    // 7) 提交所有下载任务（下载器内部使用 .tmp → rename 原子覆盖）
    for (const auto& pf : pending)
        pImpl->downloader.addTask(pf.target_path.string(), pf.url, pf.md5);

    // 8) 轮询进度（监控线程自行检测完成度后退，避免 wait_done 标志竞争）
    size_t total_sz = pending.size();
    std::thread monitor([this, &progress, total, total_sz]() {
        while (true)
        {
            size_t completed = pImpl->downloader.getCompletedCount();
            size_t failed    = pImpl->downloader.getFailed().size();
            size_t success   = (completed > failed) ? (completed - failed) : 0;

            progress.set_value(static_cast<int>(std::min(success, static_cast<size_t>(total))));

            if (completed >= total_sz)
                break;

            std::wstring txt = L"已下载 " + std::to_wstring(success) +
                               L" / " + std::to_wstring(total);
            if (failed > 0)
                txt += L"（失败 " + std::to_wstring(failed) + L" 个）";
            progress.set_status(txt);

            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
        progress.set_value(total);
    });

    pImpl->downloader.wait();
    if(monitor.joinable())
    {
        monitor.join();
    }

    // 9) 检查失败
    auto failed_map = pImpl->downloader.getFailed();
    size_t fail_cnt = failed_map.size();

    if (fail_cnt > 0)
    {
        std::string msg = "部分文件下载失败（共 " + std::to_string(fail_cnt) + " 个）:\n";
        for (const auto& [id, err] : failed_map)
            msg += "  [任务#" + std::to_string(id) + "] (" +
                   std::to_string(err.first) + ") " + err.second + "\n";

        progress.set_status(L"下载完成，部分文件失败");
        std::this_thread::sleep_for(std::chrono::seconds(1));
        progress.close();
        throw network_error(msg);
    }

    // 10) 更新本地 dependents.json
    pImpl->local_dependents_json = pImpl->remote_dependents_json;
    std::ofstream ofs(pImpl->dependents_json_path / "dependents.json");
    if (ofs.is_open()) ofs << pImpl->remote_dependents_json.dump(4);

    progress.set_status(L"下载完成");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    progress.close();
    return true;
}

GIMapDownloader::GIMapDownloader()
    : pImpl(std::make_unique<GIMapDownloaderImpl>())
{
}