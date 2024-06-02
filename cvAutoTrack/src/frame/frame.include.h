#pragma once
#include <opencv2/opencv.hpp>

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
        virtual bool initialization() { return false; }
        virtual bool uninitialized() { return false; }
        virtual bool get_frame(cv::Mat& frame) = 0;
        virtual bool set_capture_handle(HWND handle) = 0;
        virtual bool set_local_frame(cv::Mat frame) = 0;
        virtual bool set_local_file(std::string file) = 0;
        virtual bool set_source_handle_callback(std::function<HWND()> callback) = 0;
        virtual bool set_source_frame_callback(std::function<cv::Mat()> callback) = 0;
        virtual bool set_frame_rect_callback(std::function<cv::Rect(cv::Rect)> callback) = 0;
    };

} // namespace tianli::frame
