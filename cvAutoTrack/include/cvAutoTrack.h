#ifndef CVAUTOTRACE_H
#define CVAUTOTRACE_H

#if defined(_WIN32) || defined(_WIN64) || defined(_WIN128) || defined(__CYGWIN__)
#ifdef CVAUTOTRACK_EXPORTS
#define CVAUTOTRACK_PORT __declspec(dllexport)
#else
#define CVAUTOTRACK_PORT __declspec(dllimport)
#endif
#define CVAUTOTRACK_API CVAUTOTRACK_PORT
#elif __GNUC__ >= 4
#define CVAUTOTRACK_API __attribute__((visibility("default")))
#else
#define CVAUTOTRACK_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    //__declspec(deprecated("** this is a deprecated function, your should used GetCompileVersion**"))
    bool CVAUTOTRACK_API verison(char* versionBuff);

    /**
     * @brief 初始化
     * @return 是否成功
     */
    bool CVAUTOTRACK_API init();

    /**
     * @brief 卸载
     * @return 是否成功
     */
    bool CVAUTOTRACK_API uninit();

    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API startServe();

    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API stopServe();

    //__declspec(deprecated("** Bitblt Mode is deprecated **"))
    bool CVAUTOTRACK_API SetUseBitbltCaptureMode();
    bool CVAUTOTRACK_API SetUseDx11CaptureMode();

    /**
     * @brief 设置窗口句柄
     * @param handle 句柄
     * @return 是否成功
     */
    bool CVAUTOTRACK_API SetHandle(long long int handle);

    /**
     * @brief 覆盖世界坐标中心设置
     * @param x x坐标
     * @param y y坐标
     * @return 是否成功
     */
    bool CVAUTOTRACK_API SetWorldCenter(double x, double y);

    /**
     * @brief 覆盖世界坐标缩放
     * @param scale 缩放
     * @return 是否成功
     */
    bool CVAUTOTRACK_API SetWorldScale(double scale);

    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API ImportMapBlock(int id_x, int id_y, const char* image_data, int image_data_size, int image_width, int image_height);
    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API ImportMapBlockCenter(int x, int y);
    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API ImportMapBlockCenterScale(int x, int y, double scale);

    /**
     * @brief 获取当前的坐标
     * @param x x坐标
     * @param y y坐标
     * @param a 角度
     * @param mapId 地图id
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetTransformOfMap(double& x, double& y, double& a, int& mapId);

    /**
     * @brief 获取当前的坐标
     * @param x x坐标
     * @param y y坐标
     * @param mapId 地图id
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetPositionOfMap(double& x, double& y, int& mapId);

    /**
     * @brief 获取玩家朝向
     * @param a 角度
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetDirection(double& a);

    /**
     * @brief 获取相机角度
     * @param a 角度
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetRotation(double& a);

    //__declspec(deprecated("**Auto Star is Deprecated**"))
    bool CVAUTOTRACK_API GetStar(double& x, double& y, bool& isEnd);
    //__declspec(deprecated("**Auto Star is Deprecated**"))
    bool CVAUTOTRACK_API GetStarJson(char* jsonBuff);

    /**
     * @brief 获取玩家UID
     * @param uid
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetUID(int& uid);

    /**
     * @brief 获取所有信息
     * @param x x坐标
     * @param y y坐标
     * @param mapId 地图id
     * @param a 玩家朝向
     * @param r 相机朝向
     * @param uid 玩家uid
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetAllInfo(double& x, double& y, int& mapId, double& a, double& r, int& uid);

    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API GetInfoLoadPicture(char* path, int& uid, double& x, double& y, double& a);
    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API GetInfoLoadVideo(char* path, char* pathOutFile);

    /**
     * @brief 调试截图
     * @return 是否成功
     */
    bool CVAUTOTRACK_API DebugCapture();

    /**
     * @brief 获取截图保存路径
     * @param path_buff 截图保存路径
     * @param buff_size 字符串长度
     * @return 是否成功
     */
    bool CVAUTOTRACK_API DebugCapturePath(const char* path_buff, int buff_size);

    /**
     * @brief 设置第三方dll路径
     * @param path 第三方dll路径
     * @return 设置后将会尝试加载dll，如果加载失败，将返回false
     */
    bool CVAUTOTRACK_API SetThirdPartyDllPath(const char* path, int buff_size);

    /**
     * @brief 获取最后一次错误码
     * @return 错误码
     */
    int  CVAUTOTRACK_API GetLastErr();

    /**
     * @brief 获取最后一次错误信息
     * @param msg_buff 错误信息
     * @param buff_size 字符串长度
     * @return 错误码
     */
    int  CVAUTOTRACK_API GetLastErrMsg(char* msg_buff, int buff_size);
    int  CVAUTOTRACK_API GetLastErrJson(char* json_buff, int buff_size);

    /**
     * @brief 关闭文件日志
     * @return 是否成功
     */
    bool CVAUTOTRACK_API SetDisableFileLog();
    /**
     * @brief 打开文件日志
     * @return 是否成功
     */
    bool CVAUTOTRACK_API SetEnableFileLog();

    /**
     * @brief 获取编译时的版本号
     * @param version_buff 版本号
     * @param buff_size 字符串长度
     * @return 是否成功
     */
    bool CVAUTOTRACK_API GetCompileVersion(char* version_buff, int buff_size);

    /**
     * @brief 获取编译时的时间
     * @param time_buff 编译时间
     * @param buff_size 字符串长度
     * @return
     */
    bool CVAUTOTRACK_API GetCompileTime(char* time_buff, int buff_size);

    //__declspec(deprecated("** Not implemented**"))
    bool CVAUTOTRACK_API GetMapIsEmbedded();
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif
    // 定义上下文结构体
    struct cvAutoTrackContextV1
    {
        // 开发保留接口
        bool (*DebugLoadMapImagePath)(const char*);

        // 资源管理接口
        bool (*InitResource)();
        bool (*UnInitResource)();

        // 缓存配置接口
        bool (*SetCacheConfig)(const char*, const char*, const char*, int);
        bool (*SetCoreCachePath)(const char*);
        bool (*GetCoreCachePath)(char*, int);

        // 服务控制接口
        bool (*StartServer)();
        bool (*StopServer)();
        bool (*SetServerInterval)(int);
        bool (*SetServerCallback)(void (*)(const char*, int));

        // 日志配置接口
        bool (*SetDisableFileLog)();
        bool (*SetEnableFileLog)();
        bool (*SetLogFilePath)(const char*);
        bool (*SetLogFileName)(const char*);

        // 截图模式配置接口
        bool (*SetUseBitbltCaptureMode)();
        bool (*SetUseGraphicsCaptureMode)();
        bool (*SetUseDwmCaptureMode)();
        bool (*SetUseLocalPictureMode)();
        bool (*SetUseLocalVideoMode)();

        // 采集相关接口
        bool (*SetCaptureHandle)(long long);
        bool (*SetCaptureHandleCallback)(long long (*)());
        bool (*SetScreenSourceCallback)(void (*)(const char*, int&));
        bool (*SetScreenSourceCallbackEx)(void (*)(const char*, int&, int&, int&));
        bool (*SetScreenSourceImage)(const char*, int);
        bool (*SetScreenSourceImageEx)(const char*, int, int, int);
        bool (*SetScreenClientRectCallback)(void (*)(int&, int&, int&, int&));

        // 跟踪缓存接口
        bool (*SetTrackCachePath)(const char*);
        bool (*SetTrackCacheName)(const char*);

        // 坐标配置接口
        bool (*SetWorldCenter)(double, double);
        bool (*SetWorldScale)(double);

        // 数据获取接口
        bool (*GetTransformOfMap)(double&, double&, double&, int&);
        bool (*GetPositionOfMap)(double&, double&, int&);
        bool (*GetDirection)(double&);
        bool (*GetRotation)(double&);
        bool (*GetUID)(int&);
        bool (*GetAllInfo)(double&, double&, int&, double&, double&, int&);

        // 调试接口
        bool (*DebugCapture)();
        bool (*DebugCapturePath)(const char*);

        // 错误处理接口
        int (*GetLastErr)();
        int (*GetLastErrMsg)(char*, int);
        int (*GetLastErrJson)(char*, int);

        // 版本信息接口
        bool (*GetCompileVersion)(char*, int);
        bool (*GetCompileTime)(char*, int);
        bool (*GetCoreModulePath)(char*, int);

        // + 2025-3-23 添加指定模块路径接口
        bool (*LoadDependModuleFromPath)(const char*, int);
    };

    // 定义上下文初始化函数
    CVAUTOTRACK_API cvAutoTrackContextV1* create_cvAutoTrack_context_v1();
    // 定义上下文销毁函数
    CVAUTOTRACK_API void destroy_cvAutoTrack_context_v1(cvAutoTrackContextV1* context);
#ifdef __cplusplus
}
#endif
#endif // CVAUTOTRACE_H