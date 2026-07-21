#include "pch.h"
#include "cvat_logger.h"

#include <spdlog/sinks/msvc_sink.h>

namespace cvat::log {

//=============================================================================
// 错误码追踪
//=============================================================================
static std::mutex g_err_mutex;
static std::vector<ErrItem> g_err_list;  // 最近 10 条
static const size_t MAX_ERR_ITEMS = 10;

void push_error(int code, std::string msg)
{
    std::lock_guard<std::mutex> lock(g_err_mutex);
    g_err_list.push_back({code, std::move(msg)});
    if (g_err_list.size() > MAX_ERR_ITEMS) {
        g_err_list.erase(g_err_list.begin());
    }
}

void clear_errors()
{
    std::lock_guard<std::mutex> lock(g_err_mutex);
    g_err_list.clear();
}

int last_error_code()
{
    std::lock_guard<std::mutex> lock(g_err_mutex);
    return g_err_list.empty() ? 0 : g_err_list.back().code;
}

std::string last_error_msg()
{
    std::lock_guard<std::mutex> lock(g_err_mutex);
    if (g_err_list.empty()) return "0: SUCCESS";
    auto& item = g_err_list.back();
    return std::to_string(item.code) + ": " + item.msg;
}

std::string error_list_json()
{
    std::lock_guard<std::mutex> lock(g_err_mutex);
    std::string json;
    json += "{\"errorCode\":";
    json += g_err_list.empty() ? "0" : std::to_string(g_err_list.back().code);
    json += ",\"errorList\":[";
    for (size_t i = 0; i < g_err_list.size(); ++i) {
        if (i > 0) json += ",";
        json += "{\"code\":" + std::to_string(g_err_list[i].code)
             + ",\"msg\":\"" + g_err_list[i].msg + "\"}";
    }
    json += "]}";
    return json;
}

//=============================================================================
// 日志系统
//=============================================================================
static std::shared_ptr<spdlog::logger> g_logger;

void init()
{
    if (g_logger) return;

    // 控制台 sink：默认 INFO
    auto console_sink = std::make_shared<spdlog::sinks::msvc_sink_mt>();
    console_sink->set_level(spdlog::level::info);

    // 文件 sink：默认 WARN
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "autoTrack.log", true /* truncate on open */
    );
    file_sink->set_level(spdlog::level::warn);

    spdlog::sinks_init_list sinks = { console_sink, file_sink };
    g_logger = std::make_shared<spdlog::logger>("cvAutoTrack", sinks);
    g_logger->set_level(spdlog::level::trace);  // logger 级别放行所有，sink 各自过滤

    // 格式：[时间] [级别] [线程id] [文件名:行号 函数] 消息
    g_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%s:%# %!] %v");

    spdlog::set_default_logger(g_logger);
    spdlog::flush_every(std::chrono::seconds(3));

    LOGI("cvAutoTrack logger initialized (console=INFO, file=WARN)");
}

std::shared_ptr<spdlog::logger> get()
{
    if (!g_logger) init();
    return g_logger;
}

} // namespace cvat::log
