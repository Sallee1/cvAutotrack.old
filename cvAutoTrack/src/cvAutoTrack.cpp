// cvAutoTrack.cpp : 定义 DLL 的导出函数。
//

#include "pch.h"
#include "cvAutoTrack.h"
#include "ErrorCode.h"
#include "AutoTrack.h"

#define INSTALL_DUMP(at_func)\
    INSTALL_DUMP_();\
	return at_func

bool __stdcall verison(char* versionBuff)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetVersion(versionBuff, 32));
}
bool __stdcall init()
{
	INSTALL_DUMP(AutoTrack::getInstance().init());
}
bool __stdcall uninit()
{
	INSTALL_DUMP(AutoTrack::getInstance().uninit());
}
bool __stdcall SetUseBitbltCaptureMode()
{
	INSTALL_DUMP(AutoTrack::getInstance().SetUseBitbltCaptureMode());
}
bool __stdcall SetUseDx11CaptureMode()
{
	INSTALL_DUMP(AutoTrack::getInstance().SetUseWindowGraphics());
}
bool __stdcall SetHandle(long long int handle = 0)
{
	INSTALL_DUMP(AutoTrack::getInstance().SetHandle(handle));
}
bool __stdcall SetWorldCenter(double x, double y)
{
	INSTALL_DUMP(AutoTrack::getInstance().SetWorldCenter(x, y));
}
bool __stdcall SetWorldScale(double scale)
{
	INSTALL_DUMP(AutoTrack::getInstance().SetWorldScale(scale));
}
bool __stdcall ImportMapBlock(int id_x, int id_y, const char* image_data, int image_data_size, int image_width, int image_height)
{
	INSTALL_DUMP(AutoTrack::getInstance().ImportMapBlock(id_x, id_y, image_data, image_data_size, image_width, image_height));
}
bool __stdcall ImportMapBlockCenter(int x, int y)
{
	INSTALL_DUMP(AutoTrack::getInstance().ImportMapBlockCenter(x, y));
}
bool __stdcall ImportMapBlockCenterScale(int x, int y, double scale)
{
	INSTALL_DUMP(AutoTrack::getInstance().ImportMapBlockCenterScale(x, y, scale));
}
bool __stdcall GetTransformOfMap(double& x, double& y, double& a, int& mapId)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetTransformOfMap(x, y, a, mapId));
}
bool __stdcall GetPositionOfMap(double& x, double& y, int& mapId)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetPositionOfMap(x, y, mapId));
}
bool __stdcall GetDirection(double& a)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetDirection(a));
}
bool __stdcall GetRotation(double& a)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetRotation(a));
}
bool __stdcall GetStar(double& x, double& y, bool& isEnd)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetStar(x, y, isEnd));
}
bool __stdcall GetStarJson(char* jsonBuff)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetStarJson(jsonBuff));
}
bool __stdcall GetUID(int& uid)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetUID(uid));
}
bool __stdcall GetAllInfo(double& x, double& y, int& mapId, double& a, double& r, int& uid)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetAllInfo(x, y, mapId, a, r, uid));
}
bool __stdcall GetInfoLoadPicture(char* path, int& uid, double& x, double& y, double& a)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetInfoLoadPicture(path, uid, x, y, a));
}
bool __stdcall GetInfoLoadVideo(char* path, char* pathOutFile)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetInfoLoadVideo(path, pathOutFile));
}
int __stdcall GetLastErr()
{
	INSTALL_DUMP(AutoTrack::getInstance().GetLastError());
}
int __stdcall GetLastErrMsg(char* msg_buff, int buff_size)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetLastErrMsg(msg_buff, buff_size));
}
int __stdcall GetLastErrJson(char* json_buff, int buff_size)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetLastErrJson(json_buff, buff_size));
}
bool __stdcall SetDisableFileLog()
{
	INSTALL_DUMP(AutoTrack::getInstance().SetDisableFileLog());
}
bool __stdcall SetEnableFileLog()
{
	INSTALL_DUMP(AutoTrack::getInstance().SetEnableFileLog());
}
bool __stdcall GetCompileVersion(char* version_buff, int buff_size)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetVersion(version_buff, buff_size));
}
bool __stdcall GetCompileTime(char* time_buff, int buff_size)
{
	INSTALL_DUMP(AutoTrack::getInstance().GetCompileTime(time_buff, buff_size));
}
bool __stdcall GetMapIsEmbedded()
{
	INSTALL_DUMP(AutoTrack::getInstance().GetMapIsEmbedded());
}
bool __stdcall DebugCapture()
{
	INSTALL_DUMP(AutoTrack::getInstance().DebugCapture());
}
bool __stdcall DebugCapturePath(const char* path_buff, int buff_size)
{
	INSTALL_DUMP(AutoTrack::getInstance().DebugCapturePath(path_buff, buff_size));
}
bool __stdcall startServe()
{
	INSTALL_DUMP(AutoTrack::getInstance().startServe());
}
bool __stdcall stopServe()
{
	INSTALL_DUMP(AutoTrack::getInstance().stopServe());
}