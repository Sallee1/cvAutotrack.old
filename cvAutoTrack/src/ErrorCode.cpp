#include "pch.h"
#include "ErrorCode.h"

ErrorCode& ErrorCode::getInstance()
{
    static ErrorCode instance;
    return instance;
}

ErrorCode& ErrorCode::operator=(const std::pair<int, string>& err_code_msg)
{
    const int& code = err_code_msg.first;
    const string& msg = err_code_msg.second;
    if (code == 0) {
        cvat::log::clear_errors();
    } else {
        LOGE("[code={}] {}", code, msg);
        cvat::log::push_error(code, msg);
    }
    return *this;
}

ErrorCode::operator int()
{
    return 0;
}

std::ostream& operator<<(std::ostream& os, const ErrorCode& err)
{
    os << cvat::log::error_list_json();
    return os;
}

int ErrorCode::getLastError()
{
    return 0;
}

std::string ErrorCode::getLastErrorMsg()
{
    return cvat::log::last_error_msg();
}

std::string ErrorCode::toJson()
{
    return cvat::log::error_list_json();
}
