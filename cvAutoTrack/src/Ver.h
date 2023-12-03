#pragma once
#include <string>
#include "version/Version.h"
#include "ErrorCode.h"
#ifndef err
#define err ErrorCode::getInstance()
#endif


class Ver
{
public:
	bool GetVersion(char* version_buff, int buff_size)
	{
		if (version_buff == nullptr || buff_size < 1)
		{
			err = { 291,"������Ϊ��ָ����ǻ�������СΪС��1" };
			return false;
		}
		if (TianLi::Version::build_version.size() > buff_size)
		{
			err = { 292,"��������С����" };
			return false;
		}
		strcpy_s(version_buff, buff_size, TianLi::Version::build_version.c_str());
		return true;
	}

	bool GetCompileTime(char* time_buff, int buff_size)
	{
		if (time_buff == nullptr || buff_size < 1)
		{
			err = { 291,"������Ϊ��ָ����ǻ�������СΪС��1" };
			return false;
		}
		if (TianLi::Version::build_time.size() > buff_size)
		{
			err = { 292,"��������С����" };
			return false;
		}
		strcpy_s(time_buff, buff_size, TianLi::Version::build_time.c_str());
		return true;
	}
};

#undef err