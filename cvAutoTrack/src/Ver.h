#pragma once
#include <string>
#include "version/Version.h"
#include "ErrorCode.h"
#ifndef ErrorCode::getInstance()
#define ErrorCode::getInstance() ErrorCode::getInstance()
#endif


class Ver
{
public:
	bool GetVersion(char* version_buff, int buff_size)
	{
		if (version_buff == nullptr || buff_size < 1)
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

	bool GetCompileTime(char* time_buff, int buff_size)
	{
		if (time_buff == nullptr || buff_size < 1)
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
};

#undef ErrorCode::getInstance()