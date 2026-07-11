#pragma once

namespace TianLi::Binary
{
	struct binary_resources
	{
		const unsigned char* data;
		size_t size;
	};

	binary_resources get_image_binary(const char* name);
}