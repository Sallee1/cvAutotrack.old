#include "pch.h"
#include "capture.window_graphics.h"

namespace tianli::frame::capture
{

    bool capture_window_graphics::get_frame(cv::Mat& frame)
    {
        if (this->is_callback)
            set_capture_handle(this->source_handle_callback());

        if (m_framePool == nullptr)
        {
            uninitialized();
            if (initialization() == false)
                return false;
        }

        auto resolved_format = utils::window_graphics::resolve_capture_color_format(this->source_handle, utils::window_graphics::graphics_global::get_instance().dxgi_device.get());
        if (resolved_format.dxgi_format != m_dxgiFormat)
        {
            m_pixelFormat = resolved_format.pixel_format;
            m_dxgiFormat = resolved_format.dxgi_format;
            m_is_hdr_capture = resolved_format.is_hdr;
            m_framePool.Recreate(m_device.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>(), m_pixelFormat, 2, m_lastSize);
            m_swapChain->ResizeBuffers(2, static_cast<uint32_t>(m_lastSize.Width), static_cast<uint32_t>(m_lastSize.Height), m_dxgiFormat, 0);
        }

        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame new_frame{ nullptr };

        new_frame = m_framePool.TryGetNextFrame();
        if (new_frame == nullptr)
        {
            return false;
        }

        auto frame_size = new_frame.ContentSize();
        if (frame_size.Width < 100 || frame_size.Height < 100)
        {
            //帧尺寸太小，可能是最小化状态，放弃初始化防止帧池出错
            uninitialized();
            return false;
        }

        auto& desc = utils::window_graphics::graphics_global::get_instance().desc_type;
        auto frame_surface = utils::window_graphics::GetDXGIInterfaceFromObject<ID3D11Texture2D>(new_frame.Surface());
        D3D11_TEXTURE2D_DESC frame_surface_desc{};
        frame_surface->GetDesc(&frame_surface_desc);

        if (frame_size.Width != m_lastSize.Width || frame_size.Height != m_lastSize.Height)
        {
            m_framePool.Recreate(m_device.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>(), m_pixelFormat, 2,
                frame_size);
            m_lastSize = frame_size;

            m_swapChain->ResizeBuffers(2, static_cast<uint32_t>(m_lastSize.Width), static_cast<uint32_t>(m_lastSize.Height),
                m_dxgiFormat, 0);
        }

        if (frame_surface_desc.Format != m_dxgiFormat)
        {
            m_dxgiFormat = frame_surface_desc.Format;
            m_pixelFormat = utils::window_graphics::to_directx_pixel_format(m_dxgiFormat);
            m_framePool.Recreate(m_device.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>(), m_pixelFormat, 2, frame_size);
            m_swapChain->ResizeBuffers(2, static_cast<uint32_t>(frame_size.Width), static_cast<uint32_t>(frame_size.Height), m_dxgiFormat, 0);
            return false;
        }

        desc.Width = static_cast<UINT>(frame_size.Width);
        desc.Height = static_cast<UINT>(frame_size.Height);
        desc.Format = frame_surface_desc.Format;

        winrt::com_ptr<ID3D11Texture2D> buffer_texture;
        if (FAILED(utils::window_graphics::graphics_global::get_instance().d3d_device->CreateTexture2D(&desc, nullptr, buffer_texture.put())))
            return false;

        D3D11_BOX client_box;
        bool client_box_available = utils::window_graphics::get_client_box(this->source_handle, desc.Width, desc.Height, &client_box);

        if (client_box_available)
        {
            m_d3dContext->CopySubresourceRegion(buffer_texture.get(), 0, 0, 0, 0, frame_surface.get(), 0, &client_box);
        }
        else
        {
            m_d3dContext->CopyResource(buffer_texture.get(), frame_surface.get());
        }

        if (!buffer_texture)
        {
            return false;
        }

        D3D11_MAPPED_SUBRESOURCE mapped_tex{};
        if (FAILED(m_d3dContext->Map(buffer_texture.get(), 0, D3D11_MAP_READ, 0, &mapped_tex)))
            return false;

        auto data = mapped_tex.pData;
        auto pitch = mapped_tex.RowPitch;
        if (data == nullptr)
        {
            m_d3dContext->Unmap(buffer_texture.get(), 0);
            return false;
        }

        if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT)
        {
            cv::Mat hdr_mat(frame_size.Height, frame_size.Width, CV_16FC4, data, pitch);
            if (client_box_available)
            {
                auto cw = static_cast<int32_t>(client_box.right - client_box.left);
                auto ch = static_cast<int32_t>(client_box.bottom - client_box.top);
                this->source_frame = hdr_mat(cv::Rect(0, 0, cw, ch)).clone();
            }
            else
            {
                this->source_frame = hdr_mat.clone();
            }
        }
        else
        {
            cv::Mat bgra_mat(frame_size.Height, frame_size.Width, CV_8UC4, data, pitch);
            if (client_box_available)
            {
                auto cw = static_cast<int32_t>(client_box.right - client_box.left);
                auto ch = static_cast<int32_t>(client_box.bottom - client_box.top);
                this->source_frame = bgra_mat(cv::Rect(0, 0, cw, ch)).clone();
            }
            else
            {
                this->source_frame = bgra_mat.clone();
            }
        }
        m_d3dContext->Unmap(buffer_texture.get(), 0);

        if (this->source_frame.empty())
            return false;
        if (this->source_frame.cols < 480 || this->source_frame.rows < 360)
            return false;
        frame = this->source_frame;
        return true;
    }
}
