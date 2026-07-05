#=============================================================================
# generate_version.cmake
# 在 PRE_BUILD 时生成 src/version/Version.h
# 替代原来的 build_version_before.bat，保证输出为 UTF-8 编码
#=============================================================================

# 源目录和目标目录
set(VERSION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/version")
set(VERSION_H_PATH   "${VERSION_SRC_DIR}/Version.h")
set(TAG_FILE         "${VERSION_SRC_DIR}/version_tag.tag")

# ---- 清理旧文件 ----
file(REMOVE
    "${VERSION_SRC_DIR}/Version.h"
    "${VERSION_SRC_DIR}/version.ver"
    "${VERSION_SRC_DIR}/version.branch"
    "${VERSION_SRC_DIR}/version_hash.hash"
    "${VERSION_SRC_DIR}/version_next.number"
)

# ---- 读取 version_tag.tag（格式例如 "Beta-6.5.1"） ----
if(NOT EXISTS "${TAG_FILE}")
    message(FATAL_ERROR "找不到版本标签文件: ${TAG_FILE}")
endif()

file(READ "${TAG_FILE}" TAG_CONTENT)
string(STRIP "${TAG_CONTENT}" TAG_CONTENT)

# 解析 prefix / major / minor / revision
# 格式: "Beta-6.5.1" 或 "6.5.1"
if(TAG_CONTENT MATCHES "^([A-Za-z]*)-?([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
    set(VPREFIX   "${CMAKE_MATCH_1}")
    set(VMAJOR    "${CMAKE_MATCH_2}")
    set(VMINOR    "${CMAKE_MATCH_3}")
    set(VREVISION "${CMAKE_MATCH_4}")
else()
    message(FATAL_ERROR "无法解析版本标签: ${TAG_CONTENT} （期望格式如 Beta-6.5.1 或 6.5.1）")
endif()

# revision 自增（仅 RelWithDebInfo 发布版本才修改 tag）
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    math(EXPR VREVISION "${VREVISION} + 1")
endif()

# ---- 获取 Git 分支名和 Hash ----
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE VBRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(VBRANCH STREQUAL "")
    set(VBRANCH "unknown")
endif()

execute_process(
    COMMAND git log -n1 --format=format:%h
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE VHASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(VHASH STREQUAL "")
    set(VHASH "0000000")
endif()

# ---- 构建版本字符串 ----
set(BUILD_VERSION "${VPREFIX} ${VMAJOR}.${VMINOR}.${VREVISION}-${VBRANCH}-${VHASH}")

# ---- 时间戳 ----
string(TIMESTAMP BUILD_DATE "%Y/%m/%d %a")
string(TIMESTAMP BUILD_DATETIME "%Y/%m/%d %a  %H:%M:%S.00")

# ---- 生成 Version.h ----
# 使用 file(WRITE) 确保 UTF-8 编码
set(VERSION_H_CONTENT
"#pragma once
namespace TianLi::Version
{
   const std::string version_prefix = \"${VPREFIX}\";
   const int version_major = ${VMAJOR};
   const int version_minor = ${VMINOR};
   const int version_revision = ${VREVISION};
   const std::string version_hash = \"${VHASH}\";
   const std::string build_version = \"${BUILD_VERSION}\";
#ifndef _DEBUG
   const std::string build_time = \"${BUILD_DATE}\";
#else
   const std::string build_time = \"${BUILD_DATETIME}\";
#endif
}
// 该文件自动生成，无需更改
")

file(WRITE "${VERSION_H_PATH}" "${VERSION_H_CONTENT}")
message(STATUS "Version: ${BUILD_VERSION}")

# ---- 更新 version_tag.tag（仅 RelWithDebInfo 发布版本才写回） ----
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(VPREFIX STREQUAL "")
        set(NEW_TAG "${VMAJOR}.${VMINOR}.${VREVISION}")
    else()
        set(NEW_TAG "${VPREFIX}-${VMAJOR}.${VMINOR}.${VREVISION}")
    endif()
    file(WRITE "${TAG_FILE}" "${NEW_TAG}")
endif()
