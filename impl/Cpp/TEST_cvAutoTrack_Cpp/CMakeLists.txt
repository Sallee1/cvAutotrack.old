#=============================================================================
# TEST_cvAutoTrack_Cpp — 测试控制台程序
#=============================================================================
add_executable(TEST_cvAutoTrack_Cpp
    TEST_cvAutoTrack_Cpp.cpp
)

target_include_directories(TEST_cvAutoTrack_Cpp PRIVATE
    "${CMAKE_SOURCE_DIR}/cvAutoTrack/include"
)

target_link_libraries(TEST_cvAutoTrack_Cpp PRIVATE
    cvAutoTrack
)

# 与 cvAutoTrack.dll 输出到同一目录，运行时直接能找到 DLL
set_target_properties(TEST_cvAutoTrack_Cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/cvAutoTrack/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_RELNOOPZ      "${CMAKE_BINARY_DIR}/cvAutoTrack/RelNoOpz"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG         "${CMAKE_BINARY_DIR}/cvAutoTrack/Debug"
)

target_compile_options(TEST_cvAutoTrack_Cpp PRIVATE
    /utf-8
    /Zc:char8_t-
)

# ============================================================================
# 安装规则：install 时把 DLL 和 exe 打包到一起，方便调试器使用
# ============================================================================
install(TARGETS TEST_cvAutoTrack_Cpp 
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
