#pragma once

#include "LumenOpus/error.h"

namespace LumenOpus
{
	static inline void on_setup()
	{
		checkCudaErrors(cudaSetDevice(0));
	}

	static inline void on_exit()
	{
		checkCudaErrors(cudaDeviceReset());
	}
}
