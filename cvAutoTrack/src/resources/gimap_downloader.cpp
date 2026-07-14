#include "pch.h"
#include "gimap_downloader.h"
#include "downloader/cfiledownloaderasync.h"
#include "downloader/cfiledownloader.h"
#include "utils/utils.progress.h"
#include "ErrorCode.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <cctype>
#include <nlohmann/json.hpp>
#include <openssl/md5.h>
#include <openssl/evp.h>

namespace fs = std::filesystem;
using Json = nlohmann::json;

//=============================================================================
// 文件 MD5 计算（与 cfiledownloader.h 中逻辑一致）
//=============================================================================
static std::string calcFileMD5(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file) return "";

#if OPENSSL_VERSION_MAJOR >= 3
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (!mdctx) return "";
    const EVP_MD* md = EVP_md5();
    if (!md) { EVP_MD_CTX_free(mdctx); return ""; }
    if (EVP_DigestInit_ex(mdctx, md, nullptr) != 1) { EVP_MD_CTX_free(mdctx); return ""; }

    char buffer[4096];
    while (file.read(buffer, sizeof(buffer)) || file.gcount())
        EVP_DigestUpdate(mdctx, buffer, file.gcount());

    unsigned char digest[MD5_DIGEST_LENGTH];
    unsigned int digest_len = 0;
    if (EVP_DigestFinal_ex(mdctx, digest, &digest_len) != 1) { EVP_MD_CTX_free(mdctx); return ""; }
    EVP_MD_CTX_free(mdctx);

    char md5_str[33]{};
    for (unsigned int i = 0; i < digest_len; ++i)
        sprintf_s(&md5_str[i * 2], 3, "%02x", digest[i]);
    return std::string(md5_str, 32);
#else
    MD5_CTX context;
    MD5_Init(&context);
    char buffer[4096];
    while (file.read(buffer, sizeof(buffer)) || file.gcount())
        MD5_Update(&context, buffer, file.gcount());
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5_Final(digest, &context);
    char md5_str[33]{};
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
        sprintf_s(&md5_str[i * 2], 3, "%02x", digest[i]);
    return std::string(md5_str, 32);
#endif
}

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
// 日期时间比较（用于 metadata.json 的 update_time）
// 格式: "YYYY-M-D HH:MM:SS" 或 "YYYY-MM-DD HH:MM:SS"
// 返回: -1 (t1 < t2), 0 (相等), 1 (t1 > t2)
//=============================================================================
static int compareUpdateTime(const std::string& t1, const std::string& t2)
{
    int y1 = 0, m1 = 0, d1 = 0, hh1 = 0, mm1 = 0, ss1 = 0;
    int y2 = 0, m2 = 0, d2 = 0, hh2 = 0, mm2 = 0, ss2 = 0;
    sscanf_s(t1.c_str(), "%d-%d-%d %d:%d:%d", &y1, &m1, &d1, &hh1, &mm1, &ss1);
    sscanf_s(t2.c_str(), "%d-%d-%d %d:%d:%d", &y2, &m2, &d2, &hh2, &mm2, &ss2);
    if (y1 != y2) return y1 < y2 ? -1 : 1;
    if (m1 != m2) return m1 < m2 ? -1 : 1;
    if (d1 != d2) return d1 < d2 ? -1 : 1;
    if (hh1 != hh2) return hh1 < hh2 ? -1 : 1;
    if (mm1 != mm2) return mm1 < mm2 ? -1 : 1;
    if (ss1 != ss2) return ss1 < ss2 ? -1 : 1;
    return 0;
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
    std::string remote_deps_md5; // 远程 dependents.json 的期望 MD5（来自 .md5 文件）
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

    // 用 GET 实际下载 dependents.json.md5（极小文件）来验证连通性，
    // 避免 HEAD 请求被 CDN/对象存储拦截返回 403
    {
        fs::path tmp_md5 = fs::temp_directory_path() / "gimap_host_check.md5";
        tianli::FileDownloader dl(tmp_md5.string(), base_host + "/dependents.json.md5");
        if (!dl.download()) {
            std::string err_msg = "无法连接到服务器: [" +
                                  std::to_string(dl.getLastErrorCode()) + "] " +
                                  dl.getLastErrorMsg();
            ErrorCode::getInstance() = { 7001, err_msg };
            throw network_error(err_msg);
        }
        std::error_code ec;
        fs::remove(tmp_md5, ec);
    }

    pImpl->host = base_host;
    return true;
}

//-------------------------------------------------------------------------
// download —— 增量下载
//-------------------------------------------------------------------------
bool GIMapDownloader::download()
{
    // 0) 轻量校验：下载远程 dependents.json.md5，与本地 dependents.json 摘要比对
    {
        std::string local_md5;
        fs::path local_json = pImpl->dependents_json_path / "dependents.json";
        if (fs::exists(local_json))
            local_md5 = calcFileMD5(local_json.string());

        fs::path tmp_md5 = fs::temp_directory_path() / "gimap_remote_deps.md5";
        try
        {
            tianli::FileDownloader dl(tmp_md5.string(), pImpl->host + "/dependents.json.md5");
            if (dl.download())
            {
                std::ifstream ifs(tmp_md5);
                std::string remote_md5;
                if (ifs.is_open())
                {
                    ifs >> remote_md5;
                    // 移除首尾空白
                    remote_md5.erase(0, remote_md5.find_first_not_of(" \t\r\n"));
                    remote_md5.erase(remote_md5.find_last_not_of(" \t\r\n") + 1);
                }
                ifs.close();
                fs::remove(tmp_md5);

                // 保存远程 MD5 用于后续对 dependents.json 的校验
                if (!remote_md5.empty())
                {
                    pImpl->remote_deps_md5 = remote_md5;
                }

                if (!local_md5.empty() && !remote_md5.empty() &&
                    _stricmp(local_md5.c_str(), remote_md5.c_str()) == 0)
                {
                    std::string redirect = dl.getLastRedirectUrl();
                    if (!redirect.empty())
                        std::cout << "  [dependents.json.md5] last redirect: " << redirect << std::endl;
                    // 摘要一致 → 无需任何下载和解析
                    return true;
                }
            }
            else
            {
                std::cout << "  [dependents.json.md5] download failed: "
                    << dl.getLastErrorMsg() << std::endl;
            }
        }
        catch (const std::exception& e)
        {
            std::cout << "  [dependents.json.md5] exception: " << e.what() << std::endl;
        }
        std::error_code ec;
        fs::remove(tmp_md5, ec);
    }

    // 1) 下载远程 dependents.json 到临时文件，带 MD5 校验和重试
    fs::path tmp_json = fs::temp_directory_path() / "gimap_remote_deps.json";
    {
        bool dl_ok = false;
        std::string dl_error;
        for (int attempt = 0; attempt < 4; attempt++)
        {
            if (attempt > 0)
            {
                std::error_code ec_retry;
                fs::remove(tmp_json, ec_retry);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            tianli::FileDownloader dl(tmp_json.string(), pImpl->host + "/dependents.json");
            if (dl.download())
            {
                std::string actual_md5 = calcFileMD5(tmp_json.string());
                if (iequals(actual_md5, pImpl->remote_deps_md5))
                {
                    dl_ok = true;
                    break;
                }
                dl_error = "MD5 不匹配: 期望 " + pImpl->remote_deps_md5 + ", 实际 " + actual_md5;
                {
                    std::string redirect_url = dl.getLastRedirectUrl();
                    if (!redirect_url.empty())
                        dl_error += "\n最后重定向至: " + redirect_url;
                }
            }
            else
            {
                dl_error = "[" + std::to_string(dl.getLastErrorCode()) + "] " + dl.getLastErrorMsg();
            }
        }

        if (!dl_ok)
        {
            std::error_code ec_clean;
            fs::remove(tmp_json, ec_clean);

            std::string warn_msg = "\"位置追踪\" 远程 dependents.json 下载校验失败！\n原因: " + dl_error + "\n将使用本地缓存，请检查网络后重试。";
            MessageBox(NULL, fs::u8path(warn_msg).wstring().c_str(), L"警告", MB_OK | MB_ICONWARNING);

            // 尝试加载本地缓存
            fs::path local_json = pImpl->dependents_json_path / "dependents.json";
            if (fs::exists(local_json))
            {
                std::ifstream ifs_local(local_json);
                if (ifs_local.is_open())
                {
                    try { ifs_local >> pImpl->local_dependents_json; } catch (...) {}
                }
                pImpl->remote_dependents_json = pImpl->local_dependents_json;
                if (!pImpl->remote_dependents_json.is_null())
                    return true; // 走本地缓存
            }
            {
                std::string err_msg = std::string("无法获取远程依赖列表且本地缓存不可用") +
                    "\n" + dl_error;
                ErrorCode::getInstance() = { 7002, err_msg };
                throw network_error(err_msg);
            }
        }
    }

    // 解析远程 JSON
    {
        std::ifstream ifs(tmp_json);
        if (!ifs.is_open())
        {
            std::string err_msg = "无法读取远程 dependents.json 临时文件";
            ErrorCode::getInstance() = { 7003, err_msg };
            throw network_error(err_msg);
        }
        try {
            ifs >> pImpl->remote_dependents_json;
        }
        catch (const std::exception& e) {
            std::string err_msg = "解析远程 dependents.json 失败: " + std::string(e.what());
            ErrorCode::getInstance() = { 7003, err_msg };
            throw network_error(err_msg);
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

    // 3) 校验：sumMD5 一致则无需下载（安全兜底，0 步骤已处理大部分情况）
    if (pImpl->verifyDependents())
        return true;

    // 必须有远程列表
    if (pImpl->remote_dependents_json.is_null())
    {
        std::string err_msg = "缺少远程依赖列表，请先调用 setHost";
        ErrorCode::getInstance() = { 7004, err_msg };
        throw network_error(err_msg);
    }

    auto filelist_it = pImpl->remote_dependents_json.find("filelist");
    if (filelist_it == pImpl->remote_dependents_json.end() || !filelist_it->is_array())
    {
        std::string err_msg = "远程 dependents.json 缺少 filelist 字段或格式错误";
        ErrorCode::getInstance() = { 7005, err_msg };
        throw network_error(err_msg);
    }

    // 确定下载根目录
    if (pImpl->local_path.empty())
        pImpl->local_path = pImpl->dependents_json_path;

    // 4) metadata.json 特殊处理：下载 + MD5 校验 + update_time 比对 + 重试 + 回退
    bool metadata_handled = false; // 已在本次流程中下载成功
    bool metadata_skipped = false; // 已是最新，无需加入 pending
    for (const auto& entry : *filelist_it)
    {
        if (entry["filename"] != "metadata.json") continue;

        std::string meta_md5 = entry.value("md5", "");
        std::string meta_url = entry["url"].get<std::string>();
        fs::path meta_target = pImpl->local_path / "metadata.json";
        fs::path tmp_meta = fs::temp_directory_path() / "gimap_remote_meta.json";

        // 读取本地 metadata.json 的 update_time
        std::string local_update_time;
        {
            std::ifstream ifs_local(meta_target);
            if (ifs_local.is_open())
            {
                Json local_meta;
                try { ifs_local >> local_meta; local_update_time = local_meta.value("update_time", ""); } catch (...) {}
            }
        }

        // 下载 + 验证 metadata.json（最多 4 次 = 首次 + 3 次重试）
        bool meta_ok = false;
        std::string remote_update_time;
        std::string meta_error;

        for (int attempt = 0; attempt < 4; attempt++)
        {
            if (attempt > 0)
            {
                std::error_code ec_retry;
                fs::remove(tmp_meta, ec_retry);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            tianli::FileDownloader dl(tmp_meta.string(), meta_url, meta_md5);
            if (dl.download())
            {
                // 解析 update_time
                Json remote_meta;
                {
                    std::ifstream ifs_rmeta(tmp_meta);
                    if (ifs_rmeta.is_open()) try { ifs_rmeta >> remote_meta; } catch (...) {}
                }
                remote_update_time = remote_meta.value("update_time", "");

                if (remote_update_time.empty())
                {
                    meta_error = "无法解析远程 metadata.json 的 update_time";
                    {
                        std::string redirect_url = dl.getLastRedirectUrl();
                        if (!redirect_url.empty())
                            meta_error += "\n最后重定向至: " + redirect_url;
                    }
                    continue;
                }

                // 与本地 update_time 比对
                if (!local_update_time.empty())
                {
                    int cmp = compareUpdateTime(remote_update_time, local_update_time);
                    if (cmp < 0)
                    {
                        meta_error = "远程版本 (" + remote_update_time + ") 早于本地 (" + local_update_time + ")，CDN 缓存未刷新";
                        {
                            std::string redirect_url = dl.getLastRedirectUrl();
                            if (!redirect_url.empty())
                                meta_error += "\n最后重定向至: " + redirect_url;
                        }
                        continue;
                    }
                    else if (cmp == 0)
                    {
                        // 日期一致 → 文件相同，无需重下
                        if (fs::exists(meta_target))
                        {
                            std::error_code ec_clean;
                            fs::remove(tmp_meta, ec_clean);
                            metadata_skipped = true;
                            meta_ok = true;
                            break;
                        }
                    }
                }

                // 远程版本比本地新（或本地不存在），接受
                meta_ok = true;
                break;
            }
            else
            {
                meta_error = dl.getLastErrorMsg();
            }
        }

        if (meta_ok)
        {
            if (!metadata_skipped)
            {
                // 将下载好的 metadata.json 移动到目标位置
                std::error_code ec_rename;
                fs::create_directories(meta_target.parent_path());
                fs::rename(tmp_meta, meta_target, ec_rename);
                if (ec_rename)
                {
                    std::error_code ec_clean;
                    fs::remove(tmp_meta, ec_clean);
                    std::string err_msg = "移动 metadata.json 失败: " + ec_rename.message();
                    ErrorCode::getInstance() = { 7006, err_msg };
                    throw network_error(err_msg);
                }
                metadata_handled = true;
            }
        }
        else
        {
            std::error_code ec_clean;
            fs::remove(tmp_meta, ec_clean);

            std::string warn_msg = "\"位置追踪\" metadata.json 下载或验证失败！\n原因: " + meta_error + "\n将使用本地缓存，请检查网络后重试。";
            MessageBox(NULL, fs::u8path(warn_msg).wstring().c_str(), L"警告", MB_OK | MB_ICONWARNING);

            if (fs::exists(meta_target))
            {
                // 本地有缓存，回退
                return true;
            }
            {
                std::string err_msg = std::string("metadata.json 下载失败且本地无可用缓存") +
                    "\n" + meta_error;
                ErrorCode::getInstance() = { 7006, err_msg };
                throw network_error(err_msg);
            }
        }
        break;
    }

    // 5) 增量筛查
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
        // metadata.json 已提前处理，跳过
        if (filename == "metadata.json" && (metadata_handled || metadata_skipped)) continue;

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

        ErrorCode::getInstance() = { 7007, msg };
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