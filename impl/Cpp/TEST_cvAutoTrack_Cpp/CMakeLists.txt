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

# ============================================================================
# 安装规则：install 时把 DLL 和 exe 打包到一起，方便调试器使用
# ============================================================================
install(TARGETS TEST_cvAutoTrack_Cpp 
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
