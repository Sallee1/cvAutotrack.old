#include "pch.h"
#include "frame.include.h"
#include "capture/capture.bitblt.h"
#include "capture/capture.window_graphics.h"

namespace tianli::frame
{
    std::shared_ptr<frame_source> frame_source:: create(source_type type)
    {
        switch (type)
        {
        case source_type::bitblt:
            return std::make_shared<capture::capture_bitblt>();
        case source_type::window_graphics:
            return std::make_shared<capture::capture_window_graphics>();
        default:
            return nullptr;
        }
    }
} // namespace tianli::frame
