project(cvAutoTrack LANGUAGES CXX C RC)

# ============================================================================
# 第三方依赖路径（可通过 -D 或 CMakePresets 覆盖）
# ============================================================================
set(OPENCV_STATIC_DIR  "C:/Packages/opencv-static"  CACHE PATH "OpenCV 静态库根目录")
set(CURL_STATIC_DIR    "C:/Packages/curl-static"    CACHE PATH "libcurl 静态库根目录")
set(OPENSSL_STATIC_DIR "C:/Packages/openssl-static" CACHE PATH "OpenSSL 静态库根目录")
set(NLOHMANN_JSON_DIR  "C:/Packages/nlohmann_json"  CACHE PATH "nlohmann-json 头文件目录")

# ============================================================================
# 编译定义
# ============================================================================
add_definitions(-D_WINDOWS -D_USRDLL)
add_definitions(-DUNICODE -D_UNICODE)

# ============================================================================
# MSVC 运行时库：静态链接（/MT Release, /MTd Debug）
# 原 vcxproj 使用 MultiThreaded，且 OpenCV 静态库也为 /MT
# ============================================================================
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ============================================================================
# RelNoOpz 自定义配置：关闭所有优化，但使用 Release 运行时库（/MT 而非 /MTd）
# 这样可以在 Debug 时无需切换 Debug 库（Windows 的 debug/release 库不通用）
# ============================================================================
set(CMAKE_CXX_FLAGS_RELNOOPZ "/Od /D NDEBUG /Zi /Ob0")
set(CMAKE_C_FLAGS_RELNOOPZ   "/Od /D NDEBUG /Zi /Ob0")
set(CMAKE_EXE_LINKER_FLAGS_RELNOOPZ    "/DEBUG")
set(CMAKE_SHARED_LINKER_FLAGS_RELNOOPZ "/DEBUG")

# ============================================================================
# 头文件目录
# ============================================================================
set(CVAT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OPENCV_STATIC_DIR}/include"
    "${CURL_STATIC_DIR}/include"
    "${OPENSSL_STATIC_DIR}/include"
    "${NLOHMANN_JSON_DIR}/include"
)

# ============================================================================
# 源文件
# ============================================================================
set(CVAT_SOURCES
    # 入口
    src/AutoTrack.cpp
    src/cvAutoTrack.cpp
    src/dllmain.cpp

    # 错误码
    src/ErrorCode.cpp

    # 过滤
    src/filter/Filter.cpp
    src/filter/kalman/Kalman.cpp
    src/filter/smooth/Smooth.cpp
    src/filter/untouched/Untouched.cpp

    # 帧捕获
    src/frame/capture/capture.window_graphics.cpp
    src/frame/frame.include.cpp

    # 原神模块
    src/genshin/genshin.handle.cpp
    src/genshin/genshin.minimap.cpp
    src/genshin/genshin.screen.cpp
    src/genshin/match/position/genshin.match.position.cpp

    # 匹配
    src/match/diff/DiffMatch.cpp
    src/match/IMatcher.cpp
    src/match/matcher_impl/AkazeMatcher.cpp
    src/match/matcher_impl/FAST_SURFMatcher.cpp
    src/match/matcher_impl/SURFMatcher.cpp
    src/match/tracking.cpp

    # 资源
    src/resources/binary/resources.binary.cpp
    src/resources/gimap_downloader.cpp
    src/resources/import/resources.import.cpp
    src/resources/KeypointsCache.cpp
    src/resources/Resources.cpp
    src/resources/resources.load.cpp

    # 工具
    src/utils/log/utils.log.cpp
    src/utils/utils.progress.cpp
    src/utils/Utils.cpp

)

# ============================================================================
# DLL 目标
# ============================================================================
add_library(cvAutoTrack SHARED ${CVAT_SOURCES})

target_include_directories(cvAutoTrack PRIVATE ${CVAT_INCLUDE_DIRS})

# 导出宏
target_compile_definitions(cvAutoTrack PRIVATE CVAUTOTRACK_EXPORTS)

# RelNoOpz 模式下注入 _CVAT_DEBUG（关闭优化，但使用 Release 运行时库）
target_compile_definitions(cvAutoTrack PRIVATE
    $<$<CONFIG:RelNoOpz>:_CVAT_DEBUG>
)

# ============================================================================
# 预编译头（pch.h）
# ============================================================================
target_precompile_headers(cvAutoTrack PRIVATE src/pch.h)

# ============================================================================
# 链接库（静态链接）
# ============================================================================
# 链接库（静态链接）
# ============================================================================
set(CVAT_LIB_DIRS
    "${OPENCV_STATIC_DIR}/staticlib"
    "${CURL_STATIC_DIR}/lib"
    "${OPENSSL_STATIC_DIR}/lib"
)

target_link_directories(cvAutoTrack PRIVATE ${CVAT_LIB_DIRS})

target_link_libraries(cvAutoTrack PRIVATE
    # OpenCV
    IlmImf            ippicvmt          ippiw
    ittnotify         libjpeg-turbo     libopenjp2
    libpng            libtiff           libwebp
    opencv_img_hash4100
    opencv_core4100   opencv_imgproc4100
    opencv_imgcodecs4100
    opencv_flann4100  opencv_features2d4100
    opencv_xfeatures2d4100
    opencv_calib3d4100  opencv_video4100

    # AVIF / 图片编解码
    avif  yuv  aom  dav1d  zlib

    # 网络
    libcurl  libssl  libcrypto

    # Windows
    dxgi  windowsapp
)

# ============================================================================
# 编译器选项
# ============================================================================
target_compile_options(cvAutoTrack PRIVATE
    /utf-8
    /Zc:char8_t-
    /fp:fast
)

target_link_options(cvAutoTrack PRIVATE
    /IGNORE:4099
    /DEBUG
)

# 屏蔽系统 vcpkg 全局集成（避免灌入 C:\vcpkg\installed\*.lib 等无关库）
set_target_properties(cvAutoTrack PROPERTIES VS_GLOBAL_VcpkgEnabled "false")

# ============================================================================
# 版本生成（PRE_BUILD）
# 使用 cmake 脚本替代 bat，避免 GBK 编码问题
# ============================================================================
set(VERSION_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/generate_version.cmake")

add_custom_command(TARGET cvAutoTrack PRE_BUILD
    COMMAND "${CMAKE_COMMAND}" -P "${VERSION_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "生成版本信息 Version.h ……"
)

# ============================================================================
# 图片二进制嵌入头文件生成（PRE_BUILD）
# 调用封装好的批处理，从 resource/ 中的图片源文件自动生成头文件
# ============================================================================
set(IMG2BIN_BAT "${CMAKE_CURRENT_SOURCE_DIR}/src/resources/binary/image_to_binary.bat")

add_custom_command(TARGET cvAutoTrack PRE_BUILD
    COMMAND "${IMG2BIN_BAT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/resources/binary"
    COMMENT "生成图片二进制嵌入头文件 ……"
)

# ============================================================================
# 安装/部署（POST_BUILD，可选）
# ============================================================================
option(CVAT_COPY_TO_GAMEDIR "编译后自动复制 DLL 到游戏目录" OFF)

if(CVAT_COPY_TO_GAMEDIR)
    set(CVAT_GAMEDIR "$ENV{USERPROFILE}/AppData/LocalLow/空荧酒馆/Map" CACHE PATH "游戏 DLL 输出目录")
    set(DEPLOY_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/deploy.ps1")

    add_custom_command(TARGET cvAutoTrack POST_BUILD
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File
            "${DEPLOY_SCRIPT}"
            -DllPath "$<TARGET_FILE:cvAutoTrack>"
            -PdbPath "$<TARGET_PDB_FILE:cvAutoTrack>"
            -GameDir "${CVAT_GAMEDIR}"
            -Config "$<CONFIG>"
        COMMENT "复制 DLL/PDB 到游戏目录 ……"
    )
endif()


install(TARGETS cvAutoTrack 
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
