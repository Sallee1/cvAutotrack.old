#include "pch.h"
#include "AutoTrack.h"
#include <thread>

#include "ErrorCode.h"
#include "resources/Resources.h"

#include "frame/frame.include.h"
#include "frame/capture/capture.bitblt.h"
#include "frame/capture/capture.dxgi.h"
#include "frame/capture/capture.window_graphics.h"

#include "filter/kalman/Kalman.h"
#include "utils/Utils.h"

#include "algorithms/algorithms.direction.h"
#include "algorithms/algorithms.rotation.h"

#include "match/match.uid.h"

#include "genshin/genshin.h"

#include "version/Version.h"
#include "match/detector/FASTFeatureDetector.h"
#include "match/algorithm/FaissIndexedMatcher.h"
#include "match/algorithm/OpenCVBFMatcher.h"
#include "match/algorithm/FlannIndexedMatcher.h"
#include "genshin/genshin.screen.h"

#include <cinttypes>

AutoTrack::AutoTrack()
{
	ErrorCode::getInstance().enableWirteFile();

	genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_window_graphics>();
	genshin_handle.config.frame_source->initialization();
	genshin_avatar_position.config.pos_filter = std::make_shared<Kalman>();

	// 构造时仅创建 matcher（轻量），初始化在工作线程中异步执行
	// 瓦片缓存生成和追踪匹配使用不同的 IMatcher 实例，避免金字塔设置污染
	// 但共享同一个索引匹配器，保证全局匹配正确

	// ===== FAST-TEBLID: FAST 提取器 + TEBLID 描述子（二值）=====
	// 索引：FaissIndexedMatcher（倒排索引），可切换 faiss_factory::hnsw / hash / ivf
	// 亮度增益：1.0（正常）+ 1.75 + 3.0 补偿亮度变化
	{
		auto fast_detector = cv::makePtr<FASTFeatureDetector>(16, true);
		auto teblid_desc = cv::xfeatures2d::TEBLID::create(5.0f, cv::xfeatures2d::TEBLID::SIZE_256_BITS);
		auto bf_matcher = std::make_shared<OpenCVBFMatcher>(cv::NORM_HAMMING, false);
		auto faiss_idx = std::make_shared<FlannIndexedMatcher>(true);

		// 瓦片 matcher：detect=FAST, desc=TEBLID, 共享索引
		auto tile_matcher = std::make_shared<IMatcher>();
		tile_matcher->setFeature2D(fast_detector, teblid_desc);
		tile_matcher->setMatchAlgorithm(bf_matcher, faiss_idx);
		m_tile_matcher = tile_matcher;

		// 追踪 matcher：同算法 + 金字塔 + 亮度增强，共享索引
		auto track_matcher = std::make_shared<IMatcher>();
		track_matcher->setFeature2D(fast_detector, teblid_desc);
		track_matcher->setMatchAlgorithm(bf_matcher, faiss_idx);
		track_matcher->setPyramidScales({ 1.0, 0.666, 0.5, 0.333 });
		track_matcher->setBrightnessGains({ 1.0, 1.75, 3.0 });
		genshin_minimap.matcher = track_matcher;
	}

	// ===== 切换索引算法 =====
	// auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::hnsw(64));  // 图索引
	// auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::ivf(256));  // 倒排索引
	// auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::hash(4));   // 哈希索引
	// auto idx = std::make_shared<FlannIndexedMatcher>(true);                    // OpenCV FLANN
}

void AutoTrack::init_matcher()
{
	TianLi::Genshin::Match::init_matcher(m_tile_matcher, genshin_minimap.matcher);
}

bool AutoTrack::init()
{
	// 已初始化完成
	if (TianLi::Genshin::Match::is_matcher_ready())
		return true;

	// 已在初始化中，不重复启动线程
	bool expected = false;
	if (!m_init_pending.compare_exchange_strong(expected, true))
		return true;

	// 后台线程执行重量级初始化（Resources::install + 特征点缓存 + LSH 网格）
	std::thread([this]() {
		try {
			init_matcher();
		}
		catch (...) {
			// 初始化异常，重置标志位允许下次 retry
		}
		m_init_pending.store(false);
	}).detach();

	return true;
}

bool AutoTrack::uninit()
{
	TianLi::Genshin::Match::uninit_matcher();
    return true;
}

bool AutoTrack::SetUseBitbltCaptureMode()
{
	if (genshin_handle.config.frame_source == nullptr)
	{
		genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_bitblt>();
		return true;
	}
	if (genshin_handle.config.frame_source->type == tianli::frame::frame_source::source_type::bitblt)
	{
		return true;
	}
	genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_bitblt>();
	return true;
}

bool AutoTrack::SetUseDx11CaptureMode()
{
	if (genshin_handle.config.frame_source == nullptr)
	{
		genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_dxgi>();
		return true;
	}
	if (genshin_handle.config.frame_source->type == tianli::frame::frame_source::source_type::dxgi)
	{
		return true;
	}
	genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_dxgi>();
	return true;
}

bool AutoTrack::SetUseWindowGraphics()
{
	if (genshin_handle.config.frame_source == nullptr)
	{
		genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_window_graphics>();
		return true;
	}
	if (genshin_handle.config.frame_source->type == tianli::frame::frame_source::source_type::window_graphics)
	{
		return true;
	}
	genshin_handle.config.frame_source = std::make_shared<tianli::frame::capture::capture_window_graphics>();
	return true;
}

bool AutoTrack::ImportMapBlock(int id_x, int id_y, const char* image_data, int image_data_size, int image_width, int image_height)
{
	if (image_data_size != image_width * image_height * 4)
	{
		ErrorCode::getInstance() = { 9001,"传入图片通道不对应" };
		return false;
	}
	auto map_block = cv::Mat(image_height, image_width, CV_8UC4, (void*)image_data, cv::Mat::AUTO_STEP);
	if (map_block.empty())
	{
		ErrorCode::getInstance() = { 9002,"传入图片为空 " };
		return false;
	}
	//Resources::getInstance().set_map_block(id_x, id_y, map_block);
	UNREFERENCED_PARAMETER(id_x);
	UNREFERENCED_PARAMETER(id_y);

	return false;
}

bool AutoTrack::ImportMapBlockCenter(int x, int y)
{
	UNREFERENCED_PARAMETER(x);
	UNREFERENCED_PARAMETER(y);
	return false;
}

bool AutoTrack::ImportMapBlockCenterScale(int x, int y, double scale)
{
	UNREFERENCED_PARAMETER(x);
	UNREFERENCED_PARAMETER(y);
	UNREFERENCED_PARAMETER(scale);
	return false;
}

bool AutoTrack::SetHandle(long long int handle)
{
	if (handle == 0)
	{
		genshin_handle.config.is_auto_find_genshin = true;
		return true;
	}
	else
	{
		genshin_handle.config.is_auto_find_genshin = false;
		genshin_handle.handle = (HWND)handle;
	}
	return IsWindow(genshin_handle.handle);
}

// [弃用] 旧版硬编码坐标参数已废弃，SetWorldCenter/SetWorldScale 当前为空操作
bool AutoTrack::SetWorldCenter(double x, double y)
{
	UNREFERENCED_PARAMETER(x);
	UNREFERENCED_PARAMETER(y);
	return true;
}

bool AutoTrack::SetWorldScale(double scale)
{
	UNREFERENCED_PARAMETER(scale);
	return true;
}

bool AutoTrack::startServe()
{
	return false;
}

bool AutoTrack::stopServe()
{
	return false;
}

bool AutoTrack::SetDisableFileLog()
{
	ErrorCode::getInstance().disableWirteFile();
	return true;
}

bool AutoTrack::SetEnableFileLog()
{
	ErrorCode::getInstance().enableWirteFile();
	return true;
}

bool AutoTrack::GetVersion(char* version_buff, int buff_size)
{
	if (version_buff == NULL || buff_size < 1)
	{
		ErrorCode::getInstance() = { 291,"缓存区为空指针或是缓存区大小为小于1" };
		return false;
	}
	if (TianLi::Version::build_version.size() > buff_size)
	{
		ErrorCode::getInstance() = { 292,"缓存区大小不足" };
		return false;
	}
	strcpy_s(version_buff, buff_size, TianLi::Version::build_version.c_str());
	return true;
}

bool AutoTrack::GetCompileTime(char* time_buff, int buff_size)
{
	if (time_buff == NULL || buff_size < 1)
	{
		ErrorCode::getInstance() = { 291,"缓存区为空指针或是缓存区大小为小于1" };
		return false;
	}
	if (TianLi::Version::build_time.size() > buff_size)
	{
		ErrorCode::getInstance() = { 292,"缓存区大小不足" };
		return false;
	}
	strcpy_s(time_buff, buff_size, TianLi::Version::build_time.c_str());
	return true;
}

bool AutoTrack::GetMapIsEmbedded()
{
	return Resources::getInstance().map_is_embedded();
}

bool AutoTrack::DebugCapture()
{
	return DebugCapturePath("Capture.png", 12);
}

bool AutoTrack::DebugCapturePath(const char* path_buff, int buff_size)
{
	if (path_buff == NULL || buff_size < 1)
	{
		ErrorCode::getInstance() = { 251,"路径缓存区为空指针或是路径缓存区大小为小于1" };
		return false;
	}

	if (genshin_screen.img_screen.empty())
	{
		ErrorCode::getInstance() = { 252,"画面为空" };
		return false;
	}

	// 准备调试用全帧图像
	cv::Mat out_info_img;
	if (genshin_screen.img_screen.depth() == CV_32F)
	{
		// HDR 路径：临时对副本做一次全帧 tone map，不污染缓存
		auto& cache = genshin_screen.hdr_cache;
		out_info_img = TianLi::Genshin::tone_map_hdr_to_sdr(genshin_screen.img_screen, cache.white_point);
	}
	else
	{
		out_info_img = genshin_screen.img_screen.clone();
	}
	switch (genshin_handle.config.frame_source->type)
	{
	case tianli::frame::frame_source::source_type::bitblt:
	{
		// 绘制miniMap Rect
		cv::rectangle(out_info_img, genshin_minimap.rect_minimap, cv::Scalar(0, 0, 255), 2);
		cv::Rect Avatar = genshin_minimap.rect_avatar;
		Avatar.x += genshin_minimap.rect_minimap.x;
		Avatar.y += genshin_minimap.rect_minimap.y;

		// 绘制avatar Rect
		cv::rectangle(out_info_img, Avatar, cv::Scalar(0, 0, 255), 2);
		// 绘制UID Rect
		cv::rectangle(out_info_img, genshin_screen.rects.uid, cv::Scalar(0, 0, 255), 2);
		break;
	}
	case tianli::frame::frame_source::source_type::window_graphics:
	{
		// 绘制miniMap Rect
		cv::rectangle(out_info_img, genshin_minimap.rect_minimap, cv::Scalar(0, 0, 255), 2);
		cv::Rect Avatar = genshin_minimap.rect_avatar;
		Avatar.x += genshin_minimap.rect_minimap.x;
		Avatar.y += genshin_minimap.rect_minimap.y;

		// 绘制avatar Rect
		cv::rectangle(out_info_img, Avatar, cv::Scalar(0, 0, 255), 2);
		// 绘制UID Rect
		cv::rectangle(out_info_img, genshin_screen.rects.uid, cv::Scalar(0, 0, 255), 2);
	}
	}

	auto last_time_stream = std::stringstream();
	last_time_stream << genshin_screen.last_time;
	std::string last_time_str = last_time_stream.str();

	cv::putText(out_info_img, last_time_str, cv::Point(out_info_img.cols / 2, out_info_img.rows / 2), 1, 1, cv::Scalar(128, 128, 128, 255), 1, 16, 0);
	auto err_msg_str = ErrorCode::getInstance().toJson();
	cv::putText(out_info_img, err_msg_str, cv::Point(0, out_info_img.rows / 2 - 100), 1, 1, cv::Scalar(128, 128, 128, 128), 1, 16, 0);

	bool rel = cv::imwrite(path_buff, out_info_img);

	if (!rel)
	{
		ErrorCode::getInstance() = { 252,std::string("保存画面失败，请检查文件路径是否合法") + std::string(path_buff) };
		return false;
	}

	return clear_error_logs();
}

bool AutoTrack::LoadDependModuleFromPath(const char* path)
{
	static std::vector<fs::path> dll_list{
		"opencv_core4100.dll",
		"opencv_imgproc4100.dll",
		"opencv_imgcodecs4100.dll",
		"opencv_dnn4100.dll",
		"opencv_flann4100.dll",
		"opencv_features2d4100.dll",
		"opencv_calib3d4100.dll",
		"opencv_video4100.dll",
		"opencv_xfeatures2d4100.dll",
	};

	fs::path dll_path = fs::u8path(path);
	if (!fs::exists(dll_path))
	{
		return false;
	}

	//检查dll文件是否存在
	for (auto& dll_name : dll_list)
	{
		if (!fs::exists(dll_path / dll_name))
		{
			return false;
		}
	}

	//设置默认DLL的搜索方式，优先用户自定义的DLL搜索路径
	SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
		| LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS);

	//设置dll路径
	std::wstring wdll_path{ reinterpret_cast<const wchar_t*>(dll_path.u16string().c_str()) };
	AddDllDirectory(wdll_path.c_str());

	return true;
}

bool AutoTrack::SetResourcePath(const char* path)
{
	UNREFERENCED_PARAMETER(path);
	return false;
}

bool AutoTrack::GetTransformOfMap(double& x, double& y, double& a, int& mapId)
{
	// 触发自动懒初始化
	init();
	if (!TianLi::Genshin::Match::is_matcher_ready())
	{
		ErrorCode::getInstance() = { 30, "匹配器尚未初始化完成" };
		return false;
	}

	double x2 = 0, y2 = 0, a2 = 0;
	int mapId2 = 0;

	if (!GetPositionOfMap(x2, y2, mapId2))
	{
		return false;
	}

	GetDirection(a2);
	x = x2;
	y = y2;
	a = a2;
	mapId = mapId2;
	return clear_error_logs();
}

bool AutoTrack::GetPosition(double& x, double& y)
{
	// 触发自动懒初始化
	init();
	if (!TianLi::Genshin::Match::is_matcher_ready())
	{
		ErrorCode::getInstance() = { 30, "匹配器尚未初始化完成" };
		return false;
	}

#ifdef _CVAT_DEBUG_LOG
    auto __begin_time = std::chrono::steady_clock::now();
#endif

    // 设定一个特殊标志位用于全图匹配，避免卡住
    static bool is_no_inertial_navigator = false;

	if (try_get_genshin_windows() == false)
	{
        is_no_inertial_navigator = true;
		return false;
	}

#ifdef _CVAT_DEBUG_LOG
    auto __capture_time_len = std::chrono::steady_clock::now() - __begin_time;
    printf("[DEBUG] 截图耗时：%lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(__capture_time_len).count());
#endif
	if (getMiniMapRefMat() == false)
	{
		//ErrorCode::getInstance() = { 1001, "获取坐标时，没有识别到paimon" };
        is_no_inertial_navigator = true;
		return false;
	}

	if (genshin_minimap.img_minimap.empty())
	{
        is_no_inertial_navigator = true;
		ErrorCode::getInstance() = { 5, "原神小地图区域为空" };
		return false;
	}

#ifdef _CVAT_DEBUG_LOG
    __begin_time = std::chrono::steady_clock::now();
#endif
	TianLi::Genshin::Match::get_avatar_position(genshin_minimap, genshin_avatar_position, is_no_inertial_navigator);

    is_no_inertial_navigator = false;

	cv::Point2d pos = genshin_avatar_position.position;
	if (!genshin_avatar_position.config.is_exist_last_match_minimap)
	{
		ErrorCode::getInstance() = { 20, "追踪失败，且没有历史信息" };
		return false;
	}
#ifdef _CVAT_DEBUG_LOG
    auto __tracking_time_len = std::chrono::steady_clock::now() - __begin_time;
    printf("[DEBUG] 匹配耗时：%lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(__tracking_time_len).count());
#endif

	x = pos.x;
	y = pos.y;

	return clear_error_logs();
}

bool AutoTrack::GetPositionOfMap(double& x, double& y, int& mapId)
{
	mapId = 0;
	bool isSuccess = GetPosition(x, y);
	if (isSuccess != true)
	{
		return false;
	}

	auto raw_pos = TianLi::Utils::ConvertSpecialMapsPosition(x, y);
	//auto raw_pos = std::pair<cv::Point2d, int>{ cv::Point2d{ x,y },0 };

	mapId = raw_pos.second;
    x = raw_pos.first.x;
    y = raw_pos.first.y;
	return clear_error_logs();
}

bool AutoTrack::GetDirection(double& a)
{
	// GetDirection 不触发截图，只对最近成功截取的画面负责
	if (!genshin_screen.is_screen_fresh || genshin_screen.img_screen.empty() || !genshin_handle.is_exist)
	{
		ErrorCode::getInstance() = { 2004, "尚未获取画面，请先调用 GetPosition" };
		return false;
	}
	if (getMiniMapRefMat() == false)
	{
		//ErrorCode::getInstance() = { 2001, "获取角色朝向时，没有识别到paimon" };
		return false;
	}
	if (genshin_minimap.rect_avatar.empty())
	{
		ErrorCode::getInstance() = { 11,"原神角色小箭头区域为空" };
		return false;
	}

	direction_calculation_config  config;
	direction_calculation(genshin_minimap.img_avatar, a, config);
	if (config.error)
	{
		ErrorCode::getInstance() = config.err;
		return false;
	}

	return clear_error_logs();
}

bool AutoTrack::GetRotation(double& a)
{
	// GetRotation 不触发截图，只对最近成功截取的画面负责
	if (!genshin_screen.is_screen_fresh || genshin_screen.img_screen.empty() || !genshin_handle.is_exist)
	{
		ErrorCode::getInstance() = { 3004, "尚未获取画面，请先调用 GetPosition" };
		return false;
	}
	if (getMiniMapRefMat() == false)
	{
		//ErrorCode::getInstance() = { 3001, "获取视角朝向时，没有识别到paimon" };
		return false;
	}

	rotation_calculation_config config;
	rotation_calculation(genshin_minimap.img_minimap, a, config);
	if (config.error)
	{
		ErrorCode::getInstance() = config.err;
		return false;
	}

	return clear_error_logs();
}

bool AutoTrack::GetStar(double& x, double& y, bool& isEnd)
{
    ErrorCode::getInstance() = { 4100, "神瞳获取功能已废弃，请不要使用GetStar接口" };
    //删除查找神瞳接口，目前始终返回false
    return false;
}

bool AutoTrack::GetStarJson(char* jsonBuff)
{
    ErrorCode::getInstance() = { 4100, "神瞳获取功能已废弃，请不要使用GetStarJson接口" };
    return false;
}

bool AutoTrack::GetUID(int& uid)
{
	// GetUID 不触发截图，只对最近成功截取的画面负责
	if (!genshin_screen.is_screen_fresh || genshin_screen.img_screen.empty() || !genshin_handle.is_exist)
	{
		ErrorCode::getInstance() = { 4004, "尚未获取画面，请先调用 GetPosition" };
		return false;
	}

	cv::Mat& giUIDRef = genshin_screen.imgs.uid_maybe;

	std::vector<cv::Mat> channels;

	split(giUIDRef, channels);

	if (genshin_handle.config.frame_source->type == tianli::frame::frame_source::source_type::bitblt)
	{
		giUIDRef = channels[3];
	}
	else
	{
		cv::cvtColor(giUIDRef, giUIDRef, cv::COLOR_RGBA2GRAY);
	}

	uid_calculation_config config;
	uid_calculation(giUIDRef, uid, config);
	if (config.error)
	{
		ErrorCode::getInstance() = config.err;
		return false;
	}

	return clear_error_logs();
}

bool AutoTrack::GetAllInfo(double& x, double& y, int& mapId, double& a, double& r, int& uid)
{
	// 触发自动懒初始化
	init();
	if (!TianLi::Genshin::Match::is_matcher_ready())
	{
		ErrorCode::getInstance() = { 30, "匹配器尚未初始化完成" };
		return false;
	}

	// x,y,mapId
	{
		if (!GetPositionOfMap(x, y, mapId))
		{
			return false;
		}
	}

	// a
	{
		direction_calculation_config  config;
		direction_calculation(genshin_minimap.img_avatar, a, config);
		if (config.error)
		{
			ErrorCode::getInstance() = config.err;
		}
	}
	// r
	{
		rotation_calculation_config config;
		rotation_calculation(genshin_minimap.img_minimap, r, config);
		if (config.error)
		{
			ErrorCode::getInstance() = config.err;
		}
	}
	cv::Mat& giUIDRef = genshin_screen.imgs.uid_maybe;
	// uid
	{
		std::vector<cv::Mat> channels;

		cv::split(giUIDRef, channels);

		if (genshin_handle.config.frame_source->type == tianli::frame::frame_source::source_type::window_graphics)
		{
			cv::cvtColor(giUIDRef, giUIDRef, cv::COLOR_RGBA2GRAY);
		}
		else
		{
			giUIDRef = channels[3];
		}

		uid_calculation_config config;
		uid_calculation(giUIDRef, uid, config);
		if (config.error)
		{
			ErrorCode::getInstance() = config.err;
		}
	}

	return clear_error_logs();
}

bool AutoTrack::GetInfoLoadPicture(char* path, int& uid, double& x, double& y, double& a)
{
	UNREFERENCED_PARAMETER(path);
	UNREFERENCED_PARAMETER(uid);
	UNREFERENCED_PARAMETER(x);
	UNREFERENCED_PARAMETER(y);
	UNREFERENCED_PARAMETER(a);
	return false;
}

bool AutoTrack::GetInfoLoadVideo(char* path, char* pathOutFile)
{
	UNREFERENCED_PARAMETER(path);
	UNREFERENCED_PARAMETER(pathOutFile);
	return false;
}

int AutoTrack::GetLastError()
{
#ifdef _DEBUG
	std::cout << ErrorCode::getInstance();
#endif
	return ErrorCode::getInstance();
}

int AutoTrack::GetLastErrMsg(char* msg_buff, int buff_size)
{
	if (msg_buff == NULL || buff_size < 1)
	{
		ErrorCode::getInstance() = { 291,"缓存区为空指针或是缓存区大小为小于1" };
		return false;
	}
	std::string msg = ErrorCode::getInstance().getLastErrorMsg();
	if (msg.size() > buff_size)
	{
		ErrorCode::getInstance() = { 292,"缓存区大小不足" };
		return false;
	}
	strcpy_s(msg_buff, buff_size, msg.c_str());
	return true;
}

int AutoTrack::GetLastErrJson(char* json_buff, int buff_size)
{
	if (json_buff == NULL || buff_size < 1)
	{
		ErrorCode::getInstance() = { 291,"缓存区为空指针或是缓存区大小为小于1" };
		return false;
	}
	std::string msg = ErrorCode::getInstance().toJson();
	if (msg.size() > buff_size)
	{
		ErrorCode::getInstance() = { 292,"缓存区大小不足" };
		return false;
	}
	strcpy_s(json_buff, buff_size, msg.c_str());
	return true;
}

bool AutoTrack::try_get_genshin_windows()
{
	if (!clear_error_logs())
	{
		ErrorCode::getInstance() = { 0, "正常退出" };
		return false;
	}
	// 开始新的截图周期，先标脏。后续截图成功后再标净
	genshin_screen.is_screen_fresh = false;
	genshin_minimap.is_minimap_fresh = false;
	if (!getGengshinImpactWnd())
	{
		ErrorCode::getInstance() = { 101, "未能找到原神窗口句柄" };
		return false;
	}
	if (!getGengshinImpactScreen())
	{
		ErrorCode::getInstance() = { 103, "获取原神画面失败" };
		genshin_handle.is_exist = false;
		return false;
	}
	return true;
}

bool AutoTrack::getGengshinImpactWnd()
{
	TianLi::Genshin::get_genshin_handle(genshin_handle);
	if (genshin_handle.handle == NULL)
	{
		ErrorCode::getInstance() = { 10,"无效句柄或指定句柄所指向窗口不存在" };
		return false;
	}

	genshin_handle.config.frame_source->set_capture_handle(genshin_handle.handle);

	return true;
}

bool AutoTrack::getGengshinImpactScreen()
{
	if (!TianLi::Genshin::get_genshin_screen(genshin_handle, genshin_screen, &genshin_minimap))
	{
		ErrorCode::getInstance() = { 433, "截图失败" };
		return false;
	}
	genshin_screen.is_screen_fresh = true;
	return true;
}

bool AutoTrack::getMiniMapRefMat()
{
	// 小地图已在 get_genshin_screen 中检测完毕，此处仅验证结果有效
	if (genshin_minimap.img_minimap.empty())
		return false;
	if (!genshin_minimap.is_minimap_fresh)
		return false;
	return true;
}


