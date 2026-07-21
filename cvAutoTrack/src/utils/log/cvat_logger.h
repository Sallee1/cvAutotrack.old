#pragma once
//=============================================================================
// cvAutoTrack 统一日志模块 (基于 spdlog)
//
// 日志宏（C++20 std::source_location 自动携带文件/行号/函数名）:
//   LOGT(...)  LOGD(...)  LOGI(...)  LOGW(...)  LOGE(...)  LOGF(...)
//
// 限流日志宏（同一调用点 ≥1s 间隔才输出）:
//   LOGT_T(...) LOGD_T(...) LOGI_T(...) LOGW_T(...) LOGE_T(...) LOGF_T(...)
//
// 错误码推送（供外部 API GetLastErrMsg/GetLastErrJson 使用）:
//   CVAT_PUSH_ERR(code, msg)
//   CVAT_CLEAR_ERR()
//=============================================================================

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <source_location>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace cvat::log {

//=============================================================================
// 错误码追踪（供外部 C API 查询）
//=============================================================================
struct ErrItem {
    int code;
    std::string msg;
};

void push_error(int code, std::string msg);
void clear_errors();
int  last_error_code();
std::string last_error_msg();
std::string error_list_json();

//=============================================================================
// 日志系统初始化 / 获取
//=============================================================================
void init();
std::shared_ptr<spdlog::logger> get();

} // namespace cvat::log

//=============================================================================
// 内部辅助宏
//=============================================================================
#define CVAT_LOG_LOC spdlog::source_loc{ \
    std::source_location::current().file_name(), \
    static_cast<int>(std::source_location::current().line()), \
    std::source_location::current().function_name() }

#define CVAT_LOG_IMPL(level, ...) \
    cvat::log::get()->log(CVAT_LOG_LOC, level, __VA_ARGS__)

//=============================================================================
// 普通日志宏
//=============================================================================
#define LOGT(...) CVAT_LOG_IMPL(spdlog::level::trace,    __VA_ARGS__)
#define LOGD(...) CVAT_LOG_IMPL(spdlog::level::debug,    __VA_ARGS__)
#define LOGI(...) CVAT_LOG_IMPL(spdlog::level::info,     __VA_ARGS__)
#define LOGW(...) CVAT_LOG_IMPL(spdlog::level::warn,     __VA_ARGS__)
#define LOGE(...) CVAT_LOG_IMPL(spdlog::level::err,      __VA_ARGS__)
#define LOGF(...) CVAT_LOG_IMPL(spdlog::level::critical, __VA_ARGS__)

//=============================================================================
// 限流日志宏 — static 局部变量实现每个调用点独立计时（≥1s 间隔）
//=============================================================================
#define CVAT_THROTTLED_LOG_IMPL(level, ...) \
    do { \
        static auto _cvat_th_last = std::chrono::steady_clock::time_point{}; \
        auto _cvat_th_now = std::chrono::steady_clock::now(); \
        if (_cvat_th_now - _cvat_th_last >= std::chrono::seconds(1)) { \
            _cvat_th_last = _cvat_th_now; \
            CVAT_LOG_IMPL(level, __VA_ARGS__); \
        } \
    } while(0)

#define LOGT_T(...) CVAT_THROTTLED_LOG_IMPL(spdlog::level::trace,    __VA_ARGS__)
#define LOGD_T(...) CVAT_THROTTLED_LOG_IMPL(spdlog::level::debug,    __VA_ARGS__)
#define LOGI_T(...) CVAT_THROTTLED_LOG_IMPL(spdlog::level::info,     __VA_ARGS__)
#define LOGW_T(...) CVAT_THROTTLED_LOG_IMPL(spdlog::level::warn,     __VA_ARGS__)
#define LOGE_T(...) CVAT_THROTTLED_LOG_IMPL(spdlog::level::err,      __VA_ARGS__)
#define LOGF_T(...) CVAT_THROTTLED_LOG_IMPL(spdlog::level::critical, __VA_ARGS__)

//=============================================================================
// 错误码推送宏（写入日志 + 推入 ErrorCode 栈供外部 API 查询）
//=============================================================================
#define CVAT_PUSH_ERR(code, msg) \
    do { \
        LOGE("[code={}] {}", (code), (msg)); \
        cvat::log::push_error((code), (msg)); \
    } while(0)

// 限流版错误码推送（高频调用点使用）
#define CVAT_PUSH_ERR_T(code, msg) \
    do { \
        LOGE_T("[code={}] {}", (code), (msg)); \
        cvat::log::push_error((code), (msg)); \
    } while(0)

// 清空错误堆栈
#define CVAT_CLEAR_ERR() \
    do { \
        cvat::log::clear_errors(); \
    } while(0)
