#include "pch.h"
#include "Resources.h"
#include "resource.h"
#include "resources/import/resources.import.h"
#include <wincodec.h>
#include "serialize.h"
#include "version/Version.h"
#include "KeypointsCache.h"
#include "resources/gimap_downloader.h"
#include "resources/map_mapper_config.h"
#include "utils/log/cvat_logger.h"

namespace
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

	void LoadImg_ID2Mat(int IDB, cv::Mat& mat, const wchar_t* format = L"PNG",bool grayscale = false)
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
		mat = cv::imdecode(buf, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

		UnlockResource(pImageFile);
		FreeResource(imageResDataHandle);
	}
}

#include "resources.load.h"
#include <match/type/MatchType.h>

Resources::Resources()
{
	IconSightTemplate = TianLi::Load::load_image("icon_sight");
	IconQuestTemplate = TianLi::Load::load_image("icon_quest");
	UID = TianLi::Load::load_image("uid_");
	UIDnumber[0] = TianLi::Load::load_image("uid0");
	UIDnumber[1] = TianLi::Load::load_image("uid1");
	UIDnumber[2] = TianLi::Load::load_image("uid2");
	UIDnumber[3] = TianLi::Load::load_image("uid3");
	UIDnumber[4] = TianLi::Load::load_image("uid4");
	UIDnumber[5] = TianLi::Load::load_image("uid5");
	UIDnumber[6] = TianLi::Load::load_image("uid6");
	UIDnumber[7] = TianLi::Load::load_image("uid7");
	UIDnumber[8] = TianLi::Load::load_image("uid8");
	UIDnumber[9] = TianLi::Load::load_image("uid9");

	// install() 不再在构造时调用，由 init_matcher() 在后台线程中触发
}

Resources::~Resources()
{
	IconSightTemplate.release();
    IconQuestTemplate.release();

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
        //调试底图配置（可选）
#ifdef _CVAT_DEBUG
        {
            if (fs::exists("gimap.jpg"))
            {
                DebugMapTemplate = cv::imread("gimap.jpg", cv::IMREAD_COLOR);
            }
        }
        //调试参数（可选
        {
            if (fs::exists("debug.json"))
            {
                auto fs = std::ifstream("debug.json");
                Json debug_json;
                debug_json << fs;
                fs.close();

                //目前只解析偏移量，用于将特征点叠图到gimap.jpg
                DebugParams.offset.x = debug_json["offset"][0];
                DebugParams.offset.y = debug_json["offset"][1];
            }
        }
#endif

		fs::path download_target = (getDllPath() / "../../CVAT_Resources_Beta").lexically_normal();
        if (!fs::exists(download_target))
        {
            fs::create_directories(download_target);
        }
        auto& gimap_downloader = GIMapDownloader::getInstance();
        try
        {
            gimap_downloader.setDependentsJsonPath(download_target);
            gimap_downloader.setHost("https://cvat-ota.cocogoat.cn/download/cvautotrack/cvat_rc_beta_v1");
            //gimap_downloader.setHost("https://cvat-ota.cocogoat.cn/download/cvautotrack/cvat_rc_beta");
            gimap_downloader.setLocalPath(download_target);
            gimap_downloader.download();
        }
        catch(const std::exception& e)
        {
            fs::path ex_what = fs::u8path(e.what());
            std::wstring lex_what = ex_what.wstring();
            std::wstring warn_info = std::wstring(L"") + L"\"位置追踪\"资源下载失败！原因:\n" + lex_what;
            MessageBox(NULL, warn_info.c_str(), L"警告", MB_OK | MB_ICONWARNING);

            // 同时写入日志文件，方便用户复制重定向地址等详细信息
            CVAT_PUSH_ERR(7000, std::string("\"位置追踪\"资源下载失败: ") + e.what());
        }

		// 加载地图映射配置
		{
			auto& mapper = TianLi::MapMapperManager::getInstance();

			// 先尝试加载文件是否存在
			bool metaLoaded = false;
			{
				std::ifstream testFile(download_target / "metadata.json");
				if (testFile.is_open())
				{
					testFile.close();
					metaLoaded = true;
				}
			}

			if (metaLoaded)
			{
				if (!mapper.loadFromDir(download_target))
				{
					if (!mapper.isVersionCompatible())
					{
						std::wstring warn_info = L"\"metadata.json\" 大版本不兼容！\n"
							L"当前 DLL 仅支持 layer_version " +
							std::to_wstring(TianLi::MapMapperManager::SUPPORTED_MAJOR_VERSION) +
							L".x\n请更新 cvAutoTrack 以兼容新的地图数据格式。";
						MessageBox(NULL, warn_info.c_str(), L"严重错误", MB_OK | MB_ICONERROR);
					}
					else
					{
						std::wstring warn_info = L"\"metadata.json\" 加载失败, 坐标映射将使用默认值!";
						MessageBox(NULL, warn_info.c_str(), L"警告", MB_OK | MB_ICONWARNING);
					}
				}
			}
		}
		//LoadImg_ID2Mat(IDB_AVIF_GIMAP, MapTemplate, L"AVIF",true);
		is_installed = true;
	}
}

void Resources::release()
{
	if (is_installed == true)
	{
		DebugMapTemplate.release();
		DebugMapTemplate = cv::Mat();
		is_installed = false;
	}
}

bool Resources::map_is_embedded()
{
	return true;
}

fs::path Resources::getDllPath()
{
    HMODULE hModule = NULL;
    // 获取当前DLL自身的句柄
    GetModuleHandleEx(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR)&Resources::getDllPath, // 传入当前函数地址
        &hModule
    );

    wchar_t path[1024];
    if (hModule != NULL && GetModuleFileName(hModule, path, MAX_PATH) > 0) {
        return fs::path(path).parent_path();
    }
    return {};
}
