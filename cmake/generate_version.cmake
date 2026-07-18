#=============================================================================
# generate_version.cmake
# 在 PRE_BUILD 时生成 src/version/Version.h
#
# 指纹方案：
#   ① SOURCE_FP = MD5(所有源文件(不含tag)的SHA256拼接)
#   ② TAG_HASH = SHA256(version_tag.tag)
#   ③ FULL_FP  = MD5(SOURCE_FP | TAG_HASH)   ← 存入缓存
#
#   FULL_FP 不变 → return()，不碰 Version.h
#   FULL_FP 变了 → 版本号自增 → 再生 Version.h → 用新 tag 重算 FULL_FP 缓存
#=============================================================================

# 目录和路径
set(VERSION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/version")
set(VERSION_H_PATH   "${VERSION_SRC_DIR}/Version.h")
set(TAG_FILE         "${VERSION_SRC_DIR}/version_tag.tag")
set(FINGERPRINT_FILE "${VERSION_SRC_DIR}/.version_fingerprint")

# ============================================================================
# 第一阶段：计算指纹
# ============================================================================

file(GLOB_RECURSE CVAT_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*"
)
list(REMOVE_ITEM CVAT_SOURCE_FILES
    "${VERSION_H_PATH}"
    "${FINGERPRINT_FILE}"
    "${TAG_FILE}"
)
list(SORT CVAT_SOURCE_FILES)

# SOURCE_FP = 所有源文件(不含tag)的 MD5
set(COMBINED_HASH "")
foreach(FILE ${CVAT_SOURCE_FILES})
    file(SHA256 "${FILE}" FILE_HASH)
    string(APPEND COMBINED_HASH "${FILE_HASH}")
endforeach()
string(MD5 SOURCE_FP "${COMBINED_HASH}")

# TAG_HASH = tag 文件的 SHA256
file(SHA256 "${TAG_FILE}" TAG_HASH)

# FULL_FP = MD5(SOURCE_FP | TAG_HASH)
string(MD5 FULL_FP "${SOURCE_FP}|${TAG_HASH}")


# ============================================================================
# 第二阶段：比对缓存
# ============================================================================

if(EXISTS "${FINGERPRINT_FILE}")
    file(READ "${FINGERPRINT_FILE}" OLD_FP)
    string(STRIP "${OLD_FP}" OLD_FP)
    if(FULL_FP STREQUAL OLD_FP)
        # 读取 tag 文件显示当前版本号，但不触发生成
        file(READ "${TAG_FILE}" TAG_CONTENT)
        string(STRIP "${TAG_CONTENT}" TAG_CONTENT)
        message(STATUS "Version tag: ${TAG_CONTENT}")
        message(STATUS "Source unchanged, skip version generation")
        return()
    endif()
endif()

message(STATUS "Source changed, regenerating Version.h ...")

# ============================================================================
# 第三阶段：读取 tag → 自增 → 生成 Version.h
# ============================================================================

if(NOT EXISTS "${TAG_FILE}")
    message(FATAL_ERROR "找不到版本标签文件: ${TAG_FILE}")
endif()
file(READ "${TAG_FILE}" TAG_CONTENT)
string(STRIP "${TAG_CONTENT}" TAG_CONTENT)

if(TAG_CONTENT MATCHES "^([A-Za-z]*)-?([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
    set(VPREFIX   "${CMAKE_MATCH_1}")
    set(VMAJOR    "${CMAKE_MATCH_2}")
    set(VMINOR    "${CMAKE_MATCH_3}")
    set(VREVISION "${CMAKE_MATCH_4}")
else()
    message(FATAL_ERROR "无法解析版本标签: ${TAG_CONTENT}")
endif()

# revision 自增
math(EXPR VREVISION "${VREVISION} + 1")

# ---- Git 信息 ----
execute_process(COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE VBRANCH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(VBRANCH STREQUAL "")
    set(VBRANCH "unknown")
endif()

execute_process(COMMAND git log -n1 --format=format:%h
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE VHASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(VHASH STREQUAL "")
    set(VHASH "0000000")
endif()

# ---- 版本字符串 ----
set(BUILD_VERSION "${VPREFIX} ${VMAJOR}.${VMINOR}.${VREVISION}-${VBRANCH}-${VHASH}")

# ---- 时间戳 ----
string(TIMESTAMP BUILD_DATE "%Y/%m/%d %a")
string(TIMESTAMP BUILD_DATETIME "%Y/%m/%d %a  %H:%M:%S.00")

# ---- 生成 Version.h ----
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

# ============================================================================
# 第四阶段：写回 tag + 更新缓存
# ============================================================================

if(VPREFIX STREQUAL "")
    set(NEW_TAG_STR "${VMAJOR}.${VMINOR}.${VREVISION}")
else()
    set(NEW_TAG_STR "${VPREFIX}-${VMAJOR}.${VMINOR}.${VREVISION}")
endif()

# 写回 tag 文件
file(WRITE "${TAG_FILE}" "${NEW_TAG_STR}")

# 用新 tag 重算 FULL_FP 并缓存
file(SHA256 "${TAG_FILE}" NEW_TAG_HASH)
string(MD5 NEW_FULL_FP "${SOURCE_FP}|${NEW_TAG_HASH}")
file(WRITE "${FINGERPRINT_FILE}" "${NEW_FULL_FP}")
