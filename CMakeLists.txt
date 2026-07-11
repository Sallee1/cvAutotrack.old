#=============================================================================
# cvAutoTrack — CMake 根构建配置
#
# 使用方式（VS Developer PowerShell，已配置 vcvarsall x64）:
#   cmake --preset release
#   cmake --build --preset release
#
#   cmake --preset relnoopz
#   cmake --build --preset relnoopz
#=============================================================================
cmake_minimum_required(VERSION 3.20)

project(solution_cvAutoTrack)

# C++相关
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# windows相关（宽字节API）
add_definitions(-D_WINDOWS -D_USRDLL)
add_definitions(-DUNICODE -D_UNICODE)

# MT链接（避免依赖本地runtime）
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")     

add_compile_options(
    /utf-8          # 强制UTF-8
    /Zc:char8_t-    # 不使用char8_t
    /Gy             # 函数级 COMDAT 打包，配合 /OPT:REF 裁剪未引用代码
)
    
add_link_options(
    /IGNORE:4099    # PDB缺失警告（不需要调试三方库）
    /DEBUG
    /OPT:REF        # 剔除未引用的 COMDAT（配合 /Gy 实现函数级裁剪）
)

# 编译优化选项
set(RELEASE_NO_OPTIMIZE OFF CACHE BOOL "关闭 Release 编译优化")

if(RELEASE_NO_OPTIMIZE)
    string(REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/O2" "/Od" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "/Ob0" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "/Ob0" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    add_compile_options(/ZI)
else()
    add_compile_options(/dynamicdeopt)   # 动态反优化，保留被优化代码的调试信息
endif()

# Release 始终生成调试信息（PDB）
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")

# 三方库
include(cmake/third_party.cmake)

add_subdirectory(cvAutoTrack)
add_subdirectory(impl/Cpp/TEST_cvAutoTrack_Cpp)
