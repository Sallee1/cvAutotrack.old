#=============================================================================
# cvAutoTrack — CMake 根构建配置
# 原 vcxproj → CMake 迁移版
#
# 使用方式（VS Developer PowerShell）:
#   cmake -B build -G "Visual Studio 18 2026" -A x64
#   cmake --build build --config Release
#=============================================================================
cmake_minimum_required(VERSION 3.20)

project(solution_cvAutoTrack)

# ---- C++ 标准 ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(cvAutoTrack)
