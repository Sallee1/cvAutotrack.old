#pragma once
#include <Windows.h>
#include <string>

namespace TianLi::Utils
{
	class Win32ProgressWindow
	{
	public:
		Win32ProgressWindow() = default;
		~Win32ProgressWindow();

		bool create(const std::wstring& title, int max_value, const std::wstring& status = L"");
		void set_range(int max_value);
		void set_value(int value);
		void set_status(const std::wstring& status);
		void close();

	private:
		static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);
		static void process_messages();
		static bool register_class();

	private:
		HWND hwnd_ = nullptr;
		HWND progress_hwnd_ = nullptr;
		HWND status_hwnd_ = nullptr;
		int max_value_ = 100;
	};
}
