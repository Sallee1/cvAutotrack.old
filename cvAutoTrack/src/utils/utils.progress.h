#pragma once
#include <Windows.h>
#include <string>
#include <thread>

// 前向声明 IProgressDialog，避免在头文件中引入庞大的 ShObjIdl.h
struct IProgressDialog;

namespace TianLi::Utils
{
	class Win32ProgressWindow
	{
	public:
		Win32ProgressWindow() = default;
		~Win32ProgressWindow();

		// 创建进度窗口（返回立即，窗口在后台 COM UI 线程运行）
		bool create(const std::wstring& title, int max_value, const std::wstring& status = L"");
		void set_range(int max_value);                       // 线程安全
		void set_value(int value);                           // 线程安全
		void set_status(const std::wstring& status);         // 线程安全
		void close();                                        // 线程安全

	private:
		IProgressDialog* dialog_ = nullptr;
		std::thread ui_thread_;
		DWORD ui_thread_id_ = 0;
		DWORD max_value_ = 0;
	};
}
