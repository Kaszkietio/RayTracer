#pragma once

#include "cuda_runtime.h"
#include <iostream>

namespace LumenOpus
{
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

	void check_cuda(
		cudaError_t result, 
		const char* const func, 
		const char* const file, 
		const int line);
}