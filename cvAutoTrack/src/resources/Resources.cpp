#include "pch.h"
#include "Resources.h"
#include "resource.h"
#include "resources/import/resources.import.h"
#include <wincodec.h>

#include "serialize.h"
#include "version/Version.h"

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

        std::fstream debug_out_file(std::to_string(IDB) + "dbg_image.png", std::ios::out | std::ios::binary);
        debug_out_file.write(reinterpret_cast<char*>(pImageFile), imageFileSize);
        debug_out_file.close();

        // 直接使用OpenCV的imdecode函数从二进制数据加载图像
        std::vector<char> buf = { (char*)pImageFile, (char*)pImageFile + imageFileSize };
        mat = cv::imdecode(buf, cv::IMREAD_UNCHANGED);

        UnlockResource(pImageFile);
        FreeResource(imageResDataHandle);
    }
}
using namespace TianLi::Resource::Utils;
#ifdef USED_BINARY_IMAGE
#include "resources.load.h"
#endif //

Resources::Resources()
{
#ifdef USED_BINARY_IMAGE
    PaimonTemplate = TianLi::Resources::Load::load_image("paimon");
    StarTemplate = TianLi::Resources::Load::load_image("star");
    MinimapCailbTemplate = TianLi::Resources::Load::load_image("cailb");
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

    LoadImg_ID2Mat(IDB_PNG_MINIMAP_CAILB, MinimapCailbTemplate);

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

    // install();
}

Resources::~Resources()
{
    PaimonTemplate.release();
    MinimapCailbTemplate.release();
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
        LoadImg_ID2Mat(IDB_JPG_GIMAP, MapTemplate, L"JPG");
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

inline void MapKeypointCache::serialize(std::string outFileName)
{
    std::ofstream ofs(outFileName, std::fstream::out | std::fstream::binary);
    Tianli::Resources::Utils::serializeStream ss(ofs);
    ss << this->bulid_time;
    ss << this->bulid_version;
    ss << this->hessian_threshold;
    ss << this->octave;
    ss << this->octave_layers;
    ss << this->extended;
    ss << this->upRight;
    ss << this->keyPoints;
    ss << this->descriptors;
    ss << this->bulid_version_end;
    ss.align();
    ofs.close();
}

inline void MapKeypointCache::deSerialize(std::string infileName)
{
    std::ifstream ifs(infileName, std::fstream::out | std::fstream::binary);
    Tianli::Resources::Utils::deSerializeStream dss(ifs);
    dss >> this->bulid_time;
    dss >> this->bulid_version;
    dss >> this->hessian_threshold;
    dss >> this->octave;
    dss >> this->octave_layers;
    dss >> this->extended;
    dss >> this->upRight;
    dss >> this->keyPoints;
    dss >> this->descriptors;
    dss >> this->bulid_version_end;
    ifs.close();
}

bool save_map_keypoint_cache(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, float hessian_threshold, int octaves, int octave_layers, bool extended, bool upright)
{
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(hessian_threshold, octaves, octave_layers, extended, upright);
    detector->detectAndCompute(Resources::getInstance().MapTemplate, cv::noArray(), keypoints, descriptors);

    std::string build_time = __DATE__ " " __TIME__;

    MapKeypointCache cache(
        build_time, TianLi::Version::build_version, hessian_threshold,
        (WORD)octaves, (WORD)octave_layers, (WORD)extended, (WORD)upright,
        keypoints, descriptors);
    std::filesystem::remove("cvAutoTrack_Cache.xml");
    cache.serialize("cvAutoTrack_Cache.xml");

    return true;
}
bool load_map_keypoint_cache(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    if (std::filesystem::exists("cvAutoTrack_Cache.xml") == false)
    {
        return false;
    }

    MapKeypointCache cache;
    try {
        cache.deSerialize("cvAutoTrack_Cache.xml");
    }
    catch (std::exception) {   //缓存损坏
        return false;
    }

    if (cache.bulid_version != TianLi::Version::build_version)    //版本不一致
        return false;

    if (cache.bulid_version != cache.bulid_version_end)    //写入不完整
        return false;

    keypoints = cache.keyPoints;
    descriptors = cache.descriptors;
    return true;
}

bool get_map_keypoint(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    if (load_map_keypoint_cache(keypoints, descriptors) == false)
    {
        return save_map_keypoint_cache(keypoints, descriptors);
    }
    else
    {
        return true;
    }
}