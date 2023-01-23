#pragma once
#include <cstdint>

namespace LumenOpus
{
#ifndef NDEBUG
	enum class DeviceType : uint64_t
	{
		CPU, GPU
	};
#endif
}
