#pragma once
#include "cuda_runtime.h"

namespace LumenOpus
{
	__builtin_align__(16) struct HitRecord
	{
		float4 p;
		float4 normal;
		uint64_t sphereId;
		float t;
	};
}

