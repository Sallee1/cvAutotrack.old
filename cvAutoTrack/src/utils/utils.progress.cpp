#include "pch.h"
#include "utils.progress.h"
#include <CommCtrl.h>

#pragma comment(lib, "Comctl32.lib")

namespace TianLi::Utils
{
	namespace
	{
		constexpr wchar_t kProgressWindowClass[] = L"cvAutoTrackProgressWindow";
	}

	Win32ProgressWindow::~Win32ProgressWindow()
	{
		close();
	}

	bool Win32ProgressWindow::register_class()
	{
		static bool is_registered = false;
		if (is_registered)
			return true;

		WNDCLASSEXW wc{};
		wc.cbSize = sizeof(wc);
		wc.lpfnWndProc = WndProc;
		wc.hInstance = GetModuleHandleW(nullptr);
		wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
		wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
		wc.lpszClassName = kProgressWindowClass;
		if (RegisterClassExW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
			return false;

		is_registered = true;
		return true;
	}

	bool Win32ProgressWindow::create(const std::wstring& title, int max_value, const std::wstring& status)
	{
		if (!register_class())
			return false;

		INITCOMMONCONTROLSEX icex{};
		icex.dwSize = sizeof(icex);
		icex.dwICC = ICC_PROGRESS_CLASS;
		InitCommonControlsEx(&icex);

		hwnd_ = CreateWindowExW(
			WS_EX_APPWINDOW | WS_EX_TOPMOST,
			kProgressWindowClass,
			title.c_str(),
			WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
			CW_USEDEFAULT, CW_USEDEFAULT, 480, 130,
			nullptr,
			nullptr,
			GetModuleHandleW(nullptr),
			nullptr);
		if (!hwnd_)
			return false;

		status_hwnd_ = CreateWindowExW(
			0, L"STATIC", status.c_str(),
			WS_VISIBLE | WS_CHILD | SS_LEFT,
			16, 14, 440, 20,
			hwnd_, nullptr, GetModuleHandleW(nullptr), nullptr);

		progress_hwnd_ = CreateWindowExW(
			0, PROGRESS_CLASSW, nullptr,
			WS_VISIBLE | WS_CHILD | PBS_SMOOTH,
			16, 44, 440, 26,
			hwnd_, nullptr, GetModuleHandleW(nullptr), nullptr);

		set_range(max_value);
		set_value(0);
		ShowWindow(hwnd_, SW_SHOW);
		UpdateWindow(hwnd_);
		process_messages();
		return progress_hwnd_ != nullptr;
	}

	void Win32ProgressWindow::set_range(int max_value)
	{
		if (!progress_hwnd_)
			return;
		max_value_ = std::max(1, max_value);
		SendMessageW(progress_hwnd_, PBM_SETRANGE32, 0, static_cast<LPARAM>(max_value_));
	}

	void Win32ProgressWindow::set_value(int value)
	{
		if (!progress_hwnd_)
			return;
		int clamped_value = (std::max)(0, (std::min)(value, max_value_));
		SendMessageW(progress_hwnd_, PBM_SETPOS, static_cast<WPARAM>(clamped_value), 0);
		process_messages();
	}

	void Win32ProgressWindow::set_status(const std::wstring& status)
	{
		if (!status_hwnd_)
			return;
		SetWindowTextW(status_hwnd_, status.c_str());
		process_messages();
	}

	void Win32ProgressWindow::close()
	{
		if (hwnd_)
		{
			DestroyWindow(hwnd_);
			hwnd_ = nullptr;
		}
		progress_hwnd_ = nullptr;
		status_hwnd_ = nullptr;
		process_messages();
	}

	void Win32ProgressWindow::process_messages()
	{
		MSG msg{};
		while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}
	}

	LRESULT CALLBACK Win32ProgressWindow::WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
	{
		switch (msg)
		{
		case WM_CLOSE:
			DestroyWindow(hwnd);
			return 0;
		default:
			break;
		}
		return DefWindowProcW(hwnd, msg, wparam, lparam);
	}
}
