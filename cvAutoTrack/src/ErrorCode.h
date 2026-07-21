#pragma once
//=============================================================================
// ErrorCode — 外部 C API 兼容层（内部代码请使用日志宏 + CVAT_PUSH_ERR）
//=============================================================================
#include "utils/log/cvat_logger.h"
#include <string>
#include <vector>

using namespace std;

class ErrorCode
{
private:
    ErrorCode() = default;

public:
    ~ErrorCode() = default;
    ErrorCode(const ErrorCode&) = delete;
    ErrorCode& operator=(const ErrorCode&) = delete;

    static ErrorCode& getInstance();

    // [内部兼容] operator= 转发到 CVAT_PUSH_ERR
    ErrorCode& operator=(const std::pair<int, std::string>& err_code_msg);

    [[deprecated("Use CVAT_PUSH_ERR macro instead")]]
    operator int();

    friend std::ostream& operator<<(std::ostream& os, const ErrorCode& err);

public:
    [[deprecated("Always returns 0. Check log output instead.")]]
    int getLastError();

    [[deprecated("Use cvat::log::last_error_msg() instead")]]
    std::string getLastErrorMsg();

    [[deprecated("Use cvat::log::error_list_json() instead")]]
    std::string toJson();
};

// clear_error_logs 适配为 CVAT_CLEAR_ERR
inline bool clear_error_logs()
{
    CVAT_CLEAR_ERR();
    return true;
}
