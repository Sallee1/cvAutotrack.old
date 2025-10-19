#include "pch.h"
#include "Resources.h"
#include "resource.h"
#include "resources/import/resources.import.h"
#include <wincodec.h>
#include "serialize.h"
#include "version/Version.h"
#include "KeypointsCache.h"
#include "resources/gimap_downloader.h"

namespace TianLi::Resource::Utils
{
	void LoadBitmap_ID2Mat(int IDB, cv::Mat& mat)
	{
		auto H_Handle = LoadBitmap(GetModuleHandleW(L"CVAUTOTRACK.dll"), MAKEINTRESOURCE(IDB));
		BITMAP bmp;
		GetObject(H_Handle, sizeof(BITMAP), &bmp);
		int nChannels = bmp.bmBitsPixel == 1 ? 1 : bmp.bmBitsPixel / 8;
		//int depth = bmp.bmBitsPixel == 1 ? 1 : 8;
		cv::Mat v_mat;
		v_mat.create(cv::Size(bmp.bmWidth, bmp.bmHeight), CV_MAKETYPE(CV_8UC3, nChannels));
		GetBitmapBits(H_Handle, bmp.bmHeight * bmp.bmWidth * nChannels, v_mat.data);
		mat = v_mat;
	}
	bool HBitmap2MatAlpha(HBITMAP& _hBmp, cv::Mat& _mat)
	{
		//BITMAP操作
		BITMAP bmp;
		GetObject(_hBmp, sizeof(BITMAP), &bmp);
		int nChannels = bmp.bmBitsPixel == 1 ? 1 : bmp.bmBitsPixel / 8;
		//int depth = bmp.bmBitsPixel == 1 ? 1 : 8;
		//mat操作
		cv::Mat v_mat;
		v_mat.create(cv::Size(bmp.bmWidth, bmp.bmHeight), CV_MAKETYPE(CV_8UC3, nChannels));
		GetBitmapBits(_hBmp, bmp.bmHeight * bmp.bmWidth * nChannels, v_mat.data);
		_mat = v_mat;
		return true;
	}

	void LoadImg_ID2Mat(int IDB, cv::Mat& mat, const wchar_t* format = L"PNG")
	{
		HMODULE hModu = GetModuleHandleW(L"CVAUTOTRACK.dll");

		if (!hModu)
			throw std::runtime_error("Get Dll Instance Fail!");

		HRSRC imageResHandle = FindResource(hModu, MAKEINTRESOURCE(IDB), format);
		if (!imageResHandle)
			throw std::runtime_error("Load Image Resource Fail!");

		HGLOBAL imageResDataHandle = LoadResource(hModu, imageResHandle);
		if (!imageResDataHandle)
			throw std::runtime_error("Load Image Resource Data Fail!");

		void* pImageFile = LockResource(imageResDataHandle);
		size_t imageFileSize = SizeofResource(hModu, imageResHandle);

		// 直接使用OpenCV的imdecode函数从二进制数据加载图像
		std::vector<uint8_t> buf = { (uint8_t*)pImageFile, (uint8_t*)pImageFile + imageFileSize };
		mat = cv::imdecode(buf, cv::IMREAD_COLOR);

		UnlockResource(pImageFile);
		FreeResource(imageResDataHandle);
	}
}
using namespace TianLi::Resource::Utils;
#ifdef USED_BINARY_IMAGE
#include "resources.load.h"
#endif //
#include <match/type/MatchType.h>

Resources::Resources()
{
#ifdef USED_BINARY_IMAGE
	PaimonTemplate = TianLi::Resources::Load::load_image("paimon");
	StarTemplate = TianLi::Resources::Load::load_image("star");
	IconSightTemplate = TianLi::Resources::Load::load_image("icon_sight");
	IconQuestTemplate = TianLi::Resources::Load::load_image("icon_quest");
	UID = TianLi::Resources::Load::load_image("uid_");
	UIDnumber[0] = TianLi::Resources::Load::load_image("uid0");
	UIDnumber[1] = TianLi::Resources::Load::load_image("uid1");
	UIDnumber[2] = TianLi::Resources::Load::load_image("uid2");
	UIDnumber[3] = TianLi::Resources::Load::load_image("uid3");
	UIDnumber[4] = TianLi::Resources::Load::load_image("uid4");
	UIDnumber[5] = TianLi::Resources::Load::load_image("uid5");
	UIDnumber[6] = TianLi::Resources::Load::load_image("uid6");
	UIDnumber[7] = TianLi::Resources::Load::load_image("uid7");
	UIDnumber[8] = TianLi::Resources::Load::load_image("uid8");
	UIDnumber[9] = TianLi::Resources::Load::load_image("uid9");

	cv::cvtColor(StarTemplate, StarTemplate, cv::COLOR_RGB2GRAY);
	cv::cvtColor(UID, UID, cv::COLOR_RGB2GRAY);
	for (int i = 0; i < 10; i++)
	{
		cv::cvtColor(UIDnumber[i], UIDnumber[i], cv::COLOR_RGB2GRAY);
	}
#endif //
	LoadBitmap_ID2Mat(IDB_BITMAP_PAIMON, PaimonTemplate);
	LoadBitmap_ID2Mat(IDB_BITMAP_STAR, StarTemplate);

	LoadImg_ID2Mat(IDB_PNG_ICON_SIGHT, IconSightTemplate);
	LoadImg_ID2Mat(IDB_PNG_ICON_QUEST, IconQuestTemplate);

	LoadBitmap_ID2Mat(IDB_BITMAP_UID_, UID);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID0, UIDnumber[0]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID1, UIDnumber[1]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID2, UIDnumber[2]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID3, UIDnumber[3]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID4, UIDnumber[4]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID5, UIDnumber[5]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID6, UIDnumber[6]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID7, UIDnumber[7]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID8, UIDnumber[8]);
	LoadBitmap_ID2Mat(IDB_BITMAP_UID9, UIDnumber[9]);

	cv::cvtColor(StarTemplate, StarTemplate, cv::COLOR_RGBA2GRAY);
	cv::cvtColor(UID, UID, cv::COLOR_RGBA2GRAY);
	for (int i = 0; i < 10; i++)
	{
		cv::cvtColor(UIDnumber[i], UIDnumber[i], cv::COLOR_RGBA2GRAY);
	}
	install();
}

Resources::~Resources()
{
	PaimonTemplate.release();
	IconSightTemplate.release();
	StarTemplate.release();

	UID.release();
	UIDnumber[0].release();
	UIDnumber[1].release();
	UIDnumber[2].release();
	UIDnumber[3].release();
	UIDnumber[4].release();
	UIDnumber[5].release();
	UIDnumber[6].release();
	UIDnumber[7].release();
	UIDnumber[8].release();
	UIDnumber[9].release();
	release();
}

Resources& Resources::getInstance()
{
	static Resources instance;
	return instance;
}

void Resources::install()
{
	if (is_installed == false)
	{
		//auto& gimap_downloader = GIMapDownloader::getInstance();
		//gimap_downloader.setHost("https://cvat-ota.cocogoat.cn/download/cvautotrack/cvat_rc_beta");

		LoadImg_ID2Mat(IDB_AVIF_GIMAP, MapTemplate, L"AVIF");
		is_installed = true;
	}
}

void Resources::release()
{
	if (is_installed == true)
	{
		MapTemplate.release();
		MapTemplate = cv::Mat();
		is_installed = false;
	}
}

bool Resources::map_is_embedded()
{
	return true;
}