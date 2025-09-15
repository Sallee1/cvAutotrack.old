#include "pch.h"
#include "genshin.handle.h"
#include "genshin.include.h"

namespace TianLi::Genshin
{
    GenshinHandle func_get_handle(HWND& in)
    {
        static GenshinHandle out;
        if (in == 0)
        {
            get_genshin_handle(out);
        }
        else
        {
            update_genshin_handle(in, out);
        }
        return out;
    }

    HWND get_cloud_window()
    {
        HWND cloud_window = NULL;
        EnumWindows(
            [](HWND hwnd, LPARAM lParam) -> BOOL {
                auto cloud_window = reinterpret_cast<HWND*>(lParam);
                wchar_t buffer[1024];
                auto style = GetWindowLongPtr(hwnd, GWL_STYLE);
                GetWindowTextW(hwnd, buffer, 1024);
                if (std::wstring(buffer) == L"云·原神" && (style & WS_EX_LAYERED))
                {
                    *cloud_window = hwnd;
                    return FALSE;
                }
                return TRUE;
            },
            reinterpret_cast<LPARAM>(&cloud_window));
        return cloud_window;
    }

    HWND FindMainWindow(DWORD processId);

    void get_genshin_handle(GenshinHandle& genshin_handle)
    {
        if (genshin_handle.config.is_auto_find_genshin)
        {
            if (genshin_handle.is_exist && IsWindow(genshin_handle.handle))
            {
                set_genshin_window_variable(genshin_handle);
                return;
            }

            HWND& giHandle = genshin_handle.handle;
            auto now_class = GenshinWindowClass::Unity;

            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if (snapshot == INVALID_HANDLE_VALUE) {
                genshin_handle.is_exist = false;
                return;
            }

            PROCESSENTRY32W pe{};
            pe.dwSize = sizeof(PROCESSENTRY32W);

            if (!Process32FirstW(snapshot, &pe)) {
                CloseHandle(snapshot);
                genshin_handle.is_exist = false;
                return;
            }

            //云原神有多个窗口，还是用老算法
            if ((giHandle = get_cloud_window()))
            {
                genshin_handle.is_exist = true;
                now_class = GenshinWindowClass::Qt;
                set_genshin_window_variable(genshin_handle);
                return;
            }

            do {
                std::wstring exeName(pe.szExeFile);
                std::transform(exeName.begin(), exeName.end(), exeName.begin(), ::towlower);
                for (auto& [genshin_window_name, genshin_window_class] : genshin_handle.config.genshin_process_list)
                {
                    std::wstring targetLower(genshin_window_name);
                    std::transform(targetLower.begin(), targetLower.end(), targetLower.begin(), ::towlower);

                    //找到了同名的进程，从进程获取窗口句柄
                    if (exeName == targetLower) {
                        HWND hWnd = FindMainWindow(pe.th32ProcessID);
                        if (hWnd) {
                            CloseHandle(snapshot);
                            genshin_handle.is_exist = true;
                            now_class = genshin_window_class;
                            giHandle = hWnd;
                        }
                    }
                }
            } while (!genshin_handle.is_exist && Process32NextW(snapshot, &pe));

            //遍历了所有进程，但未找到目标窗口
            if (!genshin_handle.is_exist)
            {
                return;
            }

            //窗口投影（源） - 云·原神
            if (now_class == GenshinWindowClass::Obs || now_class == GenshinWindowClass::Qt)
            {
                genshin_handle.config.is_force_used_no_alpha = true;
            }
            else
            {
                genshin_handle.config.is_force_used_no_alpha = false;
            }
        }
        else
        {
            genshin_handle.handle = genshin_handle.config.genshin_handle;
        }
        if (genshin_handle.handle != 0)
        {
            genshin_handle.is_exist = true;
        }
        else
        {
            genshin_handle.is_exist = false;
            return;
        }
        set_genshin_window_variable(genshin_handle);
    }

    // 根据进程ID查找其主窗口句柄
    HWND FindMainWindow(DWORD processId) {
        struct HandleData {
            DWORD processId;
            HWND hWnd;
        };

        HandleData data = { processId, nullptr };

        EnumWindows([](HWND hWnd, LPARAM lParam) -> BOOL {
            HandleData& data = *reinterpret_cast<HandleData*>(lParam);
            DWORD windowProcessId = 0;
            GetWindowThreadProcessId(hWnd, &windowProcessId);

            // 匹配进程ID，且窗口可见、有标题
            if (data.processId == windowProcessId && IsWindowVisible(hWnd) && GetWindowTextLengthW(hWnd) > 0) {
                data.hWnd = hWnd;
                return FALSE; // 找到后停止枚举
            }
            return TRUE; // 继续枚举
            }, reinterpret_cast<LPARAM>(&data));

        return data.hWnd;
    }

    void set_genshin_window_variable(GenshinHandle& genshin_handle)
    {
        // 判断窗口是否存在标题栏
        if (GetWindowLong(genshin_handle.handle, GWL_STYLE) & WS_CAPTION)
        {
            genshin_handle.is_exist_title_bar = true;
        }
        else
        {
            genshin_handle.is_exist_title_bar = false;
        }
        // 获取窗口大小
        GetWindowRect(genshin_handle.handle, &genshin_handle.rect);
        // 获取除标题栏区域大小
        GetClientRect(genshin_handle.handle, &genshin_handle.rect_client);
        // 获取缩放比例
        HMONITOR hMonitor = MonitorFromWindow(genshin_handle.handle, MONITOR_DEFAULTTONEAREST);
        UINT dpiX, dpiY;
        GetDpiForMonitor(hMonitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY);
        genshin_handle.scale = dpiX / 96.0;

        {
            int x = genshin_handle.rect_client.right - genshin_handle.rect_client.left;
            int y = genshin_handle.rect_client.bottom - genshin_handle.rect_client.top;

            double f = 1, fx = 1, fy = 1;

            if (static_cast<double>(x) / static_cast<double>(y) == 16.0 / 9.0)
            {
                genshin_handle.size_frame = cv::Size(1920, 1080);
            }
            else if (static_cast<double>(x) / static_cast<double>(y) > 16.0 / 9.0)
            {
                //高型，以宽为比例

                // x = (y * 16) / 9;
                f = y / 1080.0;
                //将giFrame缩放到1920*1080的比例
                fx = x / f;
                // 将图片缩放
                genshin_handle.size_frame = cv::Size(static_cast<int>(fx), 1080);
            }
            else if (static_cast<double>(x) / static_cast<double>(y) < 16.0 / 9.0)
            {
                //宽型，以高为比例

                // x = (y * 16) / 9;
                f = x / 1920.0;
                //将giFrame缩放到1920*1080的比例
                fy = y / f;
                // 将图片缩放
                genshin_handle.size_frame = cv::Size(1920, static_cast<int>(fy));
            }
        }
    }

    void update_genshin_handle(const HWND& old_handle, GenshinHandle& out_genshin_handle)
    {
        static unsigned char tick_count = 0;
        if (IsWindowVisible(old_handle))
        {
            if (tick_count < 30)
            {
                tick_count++;
            }
            else
            {
                tick_count = 0;
                get_genshin_handle(out_genshin_handle);
            }
        }
        else
        {
            get_genshin_handle(out_genshin_handle);
        }
        return;
    }
}