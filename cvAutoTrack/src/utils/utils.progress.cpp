#include "pch.h"
#include "utils.progress.h"
#include <ShlObj.h>   // CLSID_ProgressDialog, IID_IProgressDialog
#include <ShlGuid.h>  // CLSID_ProgressDialog (部分 SDK)
#include <comdef.h>
#include <thread>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")

// PROGDLG_MARQUEEPROGRESS 在部分 SDK 中可能未定义
#ifndef PROGDLG_MARQUEEPROGRESS
#define PROGDLG_MARQUEEPROGRESS 0x20
#endif

namespace TianLi::Utils
{
	//-------------------------------------------------------------------------
	// create_impl —— 通用创建进度对话框
	//-------------------------------------------------------------------------
	bool Win32ProgressWindow::create_impl(const std::wstring& title, const std::wstring& status, DWORD flags)
	{
		// 用事件同步：等待 UI 线程初始化完再返回
		HANDLE h_ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
		if (!h_ready)
			return false;

		// 启动 UI 线程
		ui_thread_ = std::thread([this, title, status, flags, h_ready]() {
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
				nullptr, nullptr, flags | PROGDLG_AUTOTIME, nullptr);

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

	//-------------------------------------------------------------------------
	// create —— 普通进度条
	//-------------------------------------------------------------------------
	bool Win32ProgressWindow::create(const std::wstring& title, int max_value, const std::wstring& status)
	{
		max_value_ = static_cast<DWORD>(std::max(1, max_value));
		if (!create_impl(title, status, PROGDLG_NORMAL))
			return false;
		dialog_->SetProgress(0, max_value_);
		return true;
	}

	//-------------------------------------------------------------------------
	// create_marquee —— 不定进度条
	//-------------------------------------------------------------------------
	bool Win32ProgressWindow::create_marquee(const std::wstring& title, const std::wstring& status)
	{
		max_value_ = 0;
		return create_impl(title, status, PROGDLG_MARQUEEPROGRESS);
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
		if (ui_thread_id_ != 0 && dialog_) {
			// 将 StopProgressDialog + Release 封送到 UI 线程执行（COM STA 规则）
			PostThreadMessageW(ui_thread_id_, WM_QUIT, 0, 0);
		}
		if (ui_thread_.joinable())
			ui_thread_.join();
		// UI 线程的消息循环中已通过 StopProgressDialog+Release 清理 dialog_
		dialog_ = nullptr;
	}

	Win32ProgressWindow::~Win32ProgressWindow()
	{
		close();
	}
}
