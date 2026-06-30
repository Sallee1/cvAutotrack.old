#include "pch.h"
#include "utils.progress.h"
#include <ShlObj.h>   // CLSID_ProgressDialog, IID_IProgressDialog
#include <ShlGuid.h>  // CLSID_ProgressDialog (部分 SDK)
#include <comdef.h>
#include <thread>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")

namespace TianLi::Utils
{
	//-------------------------------------------------------------------------
	// create —— 启动 UI 线程创建原生现代进度对话框
	//-------------------------------------------------------------------------
	bool Win32ProgressWindow::create(const std::wstring& title, int max_value, const std::wstring& status)
	{
		max_value_ = static_cast<DWORD>(std::max(1, max_value));

		// 用事件同步：等待 UI 线程初始化完再返回
		HANDLE h_ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
		if (!h_ready)
			return false;

		// 启动 UI 线程
		ui_thread_ = std::thread([this, title, status, h_ready]() {
			ui_thread_id_ = GetCurrentThreadId();

			// 初始化 COM（UI 线程需 STA）
			HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
			if (FAILED(hr)) {
				dialog_ = nullptr;
				SetEvent(h_ready);
				return;
			}

			// 创建进度对话框
			IProgressDialog* pDlg = nullptr;
			hr = CoCreateInstance(CLSID_ProgressDialog, nullptr,
				CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pDlg));
			if (FAILED(hr) || !pDlg) {
				CoUninitialize();
				dialog_ = nullptr;
				SetEvent(h_ready);
				return;
			}
			dialog_ = pDlg;

			// 开始对话框
			pDlg->SetTitle(title.c_str());
			pDlg->SetLine(1, status.c_str(), false, nullptr);
			pDlg->StartProgressDialog(
				nullptr, nullptr, PROGDLG_NORMAL | PROGDLG_AUTOTIME, nullptr);
			pDlg->SetProgress(0, max_value_);

			// 通知主线程创建完毕
			SetEvent(h_ready);

			// 消息循环（GetMessage 阻塞，不占用 CPU）
			MSG msg{};
			while (GetMessageW(&msg, nullptr, 0, 0) > 0)
			{
				TranslateMessage(&msg);
				DispatchMessageW(&msg);
			}

			// 释放对话框
			pDlg->StopProgressDialog();
			pDlg->Release();
			dialog_ = nullptr;
			CoUninitialize();
			});

		// 等待窗口创建完毕
		WaitForSingleObject(h_ready, INFINITE);
		CloseHandle(h_ready);

		return dialog_ != nullptr;
	}

	void Win32ProgressWindow::set_range(int /*max_value*/)
	{
		// IProgressDialog 不直接支持动态 range，忽略
	}

	void Win32ProgressWindow::set_value(int value)
	{
		if (!dialog_)
			return;
		dialog_->SetProgress(static_cast<DWORD>(std::max(0, value)), max_value_);
	}

	void Win32ProgressWindow::set_status(const std::wstring& status)
	{
		if (!dialog_)
			return;
		dialog_->SetLine(1, status.c_str(), false, nullptr);
	}

	void Win32ProgressWindow::close()
	{
		if (dialog_) {
			dialog_->StopProgressDialog();
			dialog_->Release();
			dialog_ = nullptr;
		}
		// StopProgressDialog 在调用线程上处理窗口消息，PostQuitMessage 发到的是
		// 调用线程（主线程）。显式向 UI 线程发 WM_QUIT 确保消息循环退出。
		if (ui_thread_id_ != 0)
			PostThreadMessageW(ui_thread_id_, WM_QUIT, 0, 0);
		if (ui_thread_.joinable())
			ui_thread_.join();
	}

	Win32ProgressWindow::~Win32ProgressWindow()
	{
		close();
	}
}
