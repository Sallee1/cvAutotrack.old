project(cvAutoTrack LANGUAGES CXX C RC)

set(CVAT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OPENCV_STATIC_DIR}/include"
    "${CURL_STATIC_DIR}/include"
    "${OPENSSL_STATIC_DIR}/include"
    "${NLOHMANN_JSON_DIR}/include"
)

set(CVAT_SOURCES
    src/AutoTrack.cpp
    src/cvAutoTrack.cpp
    src/dllmain.cpp
    src/ErrorCode.cpp

    src/filter/Filter.cpp
    src/filter/kalman/Kalman.cpp
    src/filter/smooth/Smooth.cpp
    src/filter/untouched/Untouched.cpp

    src/frame/capture/capture.window_graphics.cpp
    src/frame/frame.include.cpp

    src/genshin/genshin.handle.cpp
    src/genshin/genshin.minimap.cpp
    src/genshin/genshin.screen.cpp
    src/genshin/match/position/genshin.match.position.cpp

    src/match/diff/DiffMatch.cpp
    src/match/FlannIndex.cpp
    src/match/IMatcher.cpp
    src/match/matcher_impl/AkazeMatcher.cpp
    src/match/matcher_impl/ORBMatcher.cpp
    src/match/matcher_impl/FAST_TEBLIDMatcher.cpp
    src/match/matcher_impl/SURFMatcher.cpp
    src/match/matcher_impl/FAST_AkazeMatcher.cpp
    src/match/matcher_impl/FAST_SURFMatcher.cpp
    src/match/tracking.cpp

    src/resources/binary/resources.binary.h
    src/resources/binary/resources.binary.cpp
    src/resources/gimap_downloader.cpp
    src/resources/import/resources.import.cpp
    src/resources/KeypointsCache.cpp
    src/resources/Resources.cpp
    src/resources/resources.load.cpp

    src/utils/log/utils.log.cpp
    src/utils/utils.progress.cpp
    src/utils/Utils.cpp

)

# 头文件列表（用于 IDE 大纲/项目视图展示）
set(CVAT_HEADERS
    include/cvAutoTrack.h
    src/AutoTrack.h
    src/pch.h
    src/resource.h
    src/Ver.h
    src/ErrorCode.h

    src/algorithms/algorithms.direction.h
    src/algorithms/algorithms.include.h
    src/algorithms/algorithms.rotation.h

    src/downloader/cfiledownloader.h
    src/downloader/cfiledownloaderasync.h

    src/filter/Filter.h
    src/filter/kalman/Kalman.h
    src/filter/smooth/Smooth.h
    src/filter/untouched/Untouched.h

    src/frame/capture/capture.bitblt.h
    src/frame/capture/capture.dxgi.h
    src/frame/capture/capture.include.h
    src/frame/capture/capture.window_graphics.h
    src/frame/capture/utils/utils.window_graphics.h
    src/frame/capture/utils/utils.window_scale.h
    src/frame/frame.include.h

    src/genshin/genshin.h
    src/genshin/genshin.handle.h
    src/genshin/genshin.include.h
    src/genshin/genshin.minimap.h
    src/genshin/genshin.screen.h
    src/genshin/match/position/genshin.match.position.h

    src/match/diff/DiffMatch.h
    src/match/FlannIndex.h
    src/match/IMatcher.h
    src/match/InertialNavigator.h
    src/match/match.include.h
    src/match/match.uid.h
    src/match/tracking.h

    src/match/matcher_impl/AkazeMatcher.h
    src/match/matcher_impl/FAST_AKAZEMatcher.h
    src/match/matcher_impl/FAST_SURFMatcher.h
    src/match/matcher_impl/FAST_TEBLIDMatcher.h
    src/match/matcher_impl/ORBMatcher.h
    src/match/matcher_impl/SURFMatcher.h
    src/match/type/MatchType.h

    src/resources/KeypointsCache.h
    src/resources/Resources.h
    src/resources/gimap_downloader.h
    src/resources/map_mapper_config.h
    src/resources/resources.load.h
    src/resources/serialize.h
    src/resources/import/resources.import.h

    src/utils/Utils.h
    src/utils/utils.progress.h
    src/utils/log/utils.log.h
)

add_library(cvAutoTrack SHARED ${CVAT_HEADERS} ${CVAT_SOURCES})

target_include_directories(cvAutoTrack PRIVATE ${CVAT_INCLUDE_DIRS})

# 导出宏
target_compile_definitions(cvAutoTrack PRIVATE CVAUTOTRACK_EXPORTS)

# 可选日志宏
set(WITH_CVAT_DEBUG_LOG OFF CACHE BOOL "run with DEBUG_LOG")

if(WITH_CVAT_DEBUG_LOG)
    target_compile_definitions(cvAutoTrack PRIVATE _CVAT_DEBUG_LOG)
endif()

# RELEASE_NO_OPTIMIZE 模式下注入 _CVAT_DEBUG（关闭优化，增加调试宏）
if(RELEASE_NO_OPTIMIZE)
    target_compile_definitions(cvAutoTrack PRIVATE _CVAT_DEBUG)
endif()

# 预编译头
target_precompile_headers(cvAutoTrack PRIVATE src/pch.h)

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

# 屏蔽系统 vcpkg 全局集成（避免灌入 C:\vcpkg\installed\*.lib 等无关库）
set_target_properties(cvAutoTrack PROPERTIES VS_GLOBAL_VcpkgEnabled "false")

# ============================================================================
# 生成前后置脚本（PRE_BUILD，合并为一个命令避免换行问题）
# 1. 版本信息 Version.h
# 2. 图片二进制嵌入头文件
# ============================================================================
set(VERSION_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/generate_version.cmake")
set(IMG2BIN_BAT "${CMAKE_CURRENT_SOURCE_DIR}/src/resources/binary/image_to_binary.bat")

add_custom_command(TARGET cvAutoTrack PRE_BUILD
    COMMAND "${CMAKE_COMMAND}" -E echo [PRE_BUILD 1/2] Generate Version.h ...
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_BUILD_TYPE=$<CONFIG> -P "${VERSION_SCRIPT}"
    COMMAND "${CMAKE_COMMAND}" -E echo [PRE_BUILD 2/2] Generate Binary Resources ...
    COMMAND "${IMG2BIN_BAT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
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
