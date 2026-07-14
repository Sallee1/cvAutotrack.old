#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace tianli::frame
{
    class frame_source
    {
    public:
        enum class source_type
        {
            unknown,
            bitblt,
            window_graphics,
            dxgi,
        };
        enum class source_mode
        {
            handle
        };

    public:
        source_type type = source_type::unknown;
        source_mode mode = source_mode::handle;
        bool is_initialized = false;
        bool is_callback = false;
        bool has_frame_rect_callback = false;
        std::function<cv::Rect(cv::Rect)> frame_rect_callback;

    public:
        static std::shared_ptr<frame_source> create(source_type type);
    public:
        frame_source() = default;
        virtual ~frame_source() = default;
        //初始化捕获器
        virtual bool initialization() { return false; }
        //卸载捕获器
        virtual bool uninitialized() { return false; }
        //获取指定handle的帧画面，每一次循环调用一次
        virtual bool get_frame(cv::Mat& frame) = 0;
        //设置待捕获的句柄
        virtual bool set_capture_handle(HWND handle) = 0;
        virtual bool set_local_frame(cv::Mat frame) = 0;
        virtual bool set_local_file(const fs::path& file) = 0;
        //当设置句柄时，调用的回调函数
        virtual bool set_source_handle_callback(std::function<HWND()> callback) = 0;
        //当截图时，调用的回调函数
        virtual bool set_source_frame_callback(std::function<cv::Mat()> callback) = 0;
        virtual bool set_frame_rect_callback(std::function<cv::Rect(cv::Rect)> callback) = 0;
    };
} // namespace tianli::frame
