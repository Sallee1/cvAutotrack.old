#pragma once
#include "capture.include.h"
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

namespace tianli::frame::capture
{
    class capture_dxgi : public capture_source
    {
    public:
        capture_dxgi()
        {
            this->type = source_type::dxgi;
        }
        ~capture_dxgi() override
        {
            uninitialized();
        }

    public:
        bool initialization() override
        {
            if (is_initialized) return true;

            // 获取窗口句柄
            HWND hwnd = source_handle;
            if (is_callback)
                hwnd = source_handle_callback();

            if (!hwnd || !IsWindow(hwnd))
                return false;

            // 获取窗口所在显示器的设备
            HMONITOR hMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
            if (!hMonitor)
                return false;

            // 创建D3D设备和输出复制
            if (!init_dxgi_device(hMonitor))
                return false;

            is_initialized = true;
            return true;
        }

        bool uninitialized() override
        {
            if (!is_initialized)
                return true;

            // 释放DXGI资源
            if (m_d3dContext)
            {
                m_d3dContext->Flush();
                m_d3dContext.Reset();
            }
            m_d3dDevice.Reset();
            m_outputDuplication.Reset();
            m_acquiredSurface.Reset();
            m_destTexture.Reset();
            m_outputRect = { 0,0,0,0 };
            m_currentMonitor = nullptr;

            is_initialized = false;
            return true;
        }

        bool set_capture_handle(HWND handle) override
        {
            if (handle == source_handle)
                return true;

            if (!uninitialized())
                return false;

            source_handle = handle;
            if (!initialization())
                return false;

            is_callback = false;
            return true;
        }

        bool set_source_handle_callback(std::function<HWND()> callback) override
        {
            if (!callback)
                return false;

            if (!uninitialized())
                return false;

            source_handle_callback = callback;
            if (!initialization())
                return false;

            is_callback = true;
            return true;
        }

        bool get_frame(cv::Mat& frame) override
        {
            HWND hwnd = source_handle;
            if (is_callback)
                hwnd = source_handle_callback();
            if (!hwnd || !IsWindow(hwnd))
                return false;

            HMONITOR current_monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
            if (current_monitor != m_currentMonitor)
            {
                uninitialized();
                source_handle = hwnd;
                if (!initialization())
                    return false;
            }

            if (!is_initialized && !initialization())
                return false;

            if (!m_outputDuplication)
                return false;

            // 获取新帧
            IDXGIResource* pDesktopResource = nullptr;
            HRESULT hr = m_outputDuplication->AcquireNextFrame(
                0,
                &m_frameInfo,
                &pDesktopResource
            );

            if (hr == DXGI_ERROR_WAIT_TIMEOUT)
                return false;

            if (FAILED(hr))
            {
                uninitialized();
                return false;
            }

            // 将裸指针转换为 ID3D11Texture2D 并封装进 ComPtr
            ID3D11Texture2D* pTexture = nullptr;
            hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&pTexture));
            pDesktopResource->Release(); // 释放临时资源指针

            if (FAILED(hr))
            {
                m_outputDuplication->ReleaseFrame();
                m_acquiredSurface.Reset();
                return false;
            }

            m_acquiredSurface.Reset();
            m_acquiredSurface.Attach(pTexture);

            // 如果需要，调整目标纹理大小
            D3D11_TEXTURE2D_DESC acquired_desc{};
            m_acquiredSurface->GetDesc(&acquired_desc);
            if (!m_destTexture ||
                acquired_desc.Width != static_cast<UINT>(m_destSize.width) ||
                acquired_desc.Height != static_cast<UINT>(m_destSize.height) ||
                acquired_desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM)
            {
                m_destSize = cv::Size(static_cast<int>(acquired_desc.Width), static_cast<int>(acquired_desc.Height));
                if (!resize_destination_texture())
                {
                    m_outputDuplication->ReleaseFrame();
                    return false;
                }
            }

            // 复制帧数据
            m_d3dContext->CopyResource(m_destTexture.Get(), m_acquiredSurface.Get());

            // 映射纹理以读取数据
            D3D11_MAPPED_SUBRESOURCE mapped;
            hr = m_d3dContext->Map(m_destTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
            if (FAILED(hr))
            {
                m_outputDuplication->ReleaseFrame();
                return false;
            }

            // 转换为OpenCV Mat
            const BYTE* pData = static_cast<const BYTE*>(mapped.pData);
            int width = m_destSize.width;
            int height = m_destSize.height;

            // 创建临时Mat并转换格式
            cv::Mat tempMat(height, width, CV_8UC4, const_cast<BYTE*>(pData), mapped.RowPitch);

            // 将BGRA转换为BGR
            cv::cvtColor(tempMat, source_frame, cv::COLOR_BGRA2BGR);

            m_d3dContext->Unmap(m_destTexture.Get(), 0);
            m_outputDuplication->ReleaseFrame();

            if (source_frame.empty())
                return false;

            RECT client_screen_rect{};
            if (!get_client_rect_on_screen(hwnd, client_screen_rect))
                return false;

            RECT clipped_rect{};
            if (!IntersectRect(&clipped_rect, &client_screen_rect, &m_outputRect))
                return false;

            int crop_x = clipped_rect.left - m_outputRect.left;
            int crop_y = clipped_rect.top - m_outputRect.top;
            int crop_w = clipped_rect.right - clipped_rect.left;
            int crop_h = clipped_rect.bottom - clipped_rect.top;
            if (crop_w <= 0 || crop_h <= 0)
                return false;

            cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
            roi &= cv::Rect(0, 0, source_frame.cols, source_frame.rows);
            if (roi.width <= 0 || roi.height <= 0)
                return false;
            source_frame = source_frame(roi).clone();

            frame = source_frame;
            return true;
        }

    private:
        bool init_dxgi_device(HMONITOR hMonitor)
        {
            // 创建D3D设备
            D3D_FEATURE_LEVEL featureLevel;
            HRESULT hr = D3D11CreateDevice(
                nullptr,
                D3D_DRIVER_TYPE_HARDWARE,
                nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                nullptr,
                0,
                D3D11_SDK_VERSION,
                m_d3dDevice.GetAddressOf(),
                &featureLevel,
                m_d3dContext.GetAddressOf()
            );

            if (FAILED(hr))
                return false;

            // 获取 IDXGIDevice 接口
            Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
            hr = m_d3dDevice.As(&dxgiDevice);
            if (FAILED(hr))
                return false;

            // 从 IDXGIDevice 获取 IDXGIAdapter
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            hr = dxgiDevice->GetAdapter(adapter.GetAddressOf());
            if (FAILED(hr))
                return false;

            // 枚举输出
            Microsoft::WRL::ComPtr<IDXGIOutput> output;
            UINT outputIdx = 0;
            while (adapter->EnumOutputs(outputIdx, output.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
            {
                Microsoft::WRL::ComPtr<IDXGIOutput1> output1;
                hr = output->QueryInterface(__uuidof(IDXGIOutput1), reinterpret_cast<void**>(output1.GetAddressOf()));
                if (SUCCEEDED(hr))
                {
                    DXGI_OUTPUT_DESC desc;
                    output->GetDesc(&desc);

                    if (desc.Monitor == hMonitor)
                    {
                        // 创建输出复制
                        hr = output1->DuplicateOutput(m_d3dDevice.Get(), m_outputDuplication.GetAddressOf());
                        if (SUCCEEDED(hr))
                        {
                            // 获取输出尺寸
                            DXGI_OUTPUT_DESC outputDesc;
                            output->GetDesc(&outputDesc);
                            m_destSize.width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
                            m_destSize.height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
                            m_outputRect = outputDesc.DesktopCoordinates;
                            m_currentMonitor = desc.Monitor;

                            // 创建目标纹理
                            if (!create_destination_texture())
                                return false;

                            return true;
                        }
                    }
                }
                output.Reset();
                outputIdx++;
            }

            return false;
        }

        bool create_destination_texture()
        {
            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = m_destSize.width;
            desc.Height = m_destSize.height;
            desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.ArraySize = 1;
            desc.BindFlags = 0;
            desc.MiscFlags = 0;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_STAGING;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

            HRESULT hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, m_destTexture.GetAddressOf());
            return SUCCEEDED(hr);
        }

        bool resize_destination_texture()
        {
            m_destTexture.Reset();
            return create_destination_texture();
        }

        bool get_client_rect_on_screen(HWND hwnd, RECT& out_rect) const
        {
            RECT client_rect{};
            if (!GetClientRect(hwnd, &client_rect))
                return false;
            POINT pt{ 0, 0 };
            if (!ClientToScreen(hwnd, &pt))
                return false;
            out_rect.left = pt.x;
            out_rect.top = pt.y;
            out_rect.right = pt.x + client_rect.right;
            out_rect.bottom = pt.y + client_rect.bottom;
            return out_rect.right > out_rect.left && out_rect.bottom > out_rect.top;
        }

    private:
        Microsoft::WRL::ComPtr<ID3D11Device> m_d3dDevice;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_d3dContext;
        Microsoft::WRL::ComPtr<IDXGIOutputDuplication> m_outputDuplication;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_destTexture;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_acquiredSurface;

        DXGI_OUTDUPL_FRAME_INFO m_frameInfo = {};
        cv::Size m_destSize;
        RECT m_outputRect{ 0, 0, 0, 0 };
        HMONITOR m_currentMonitor = nullptr;
    };
} // namespace tianli::frame::capture
