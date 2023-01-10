#pragma once
#include "cuda_runtime.h"
#include "LumenOpus/utils.h"
#include "helper_math.h"

namespace LumenOpus
{
	__device_builtin__ __builtin_align__(16) 
	class Ray
	{
	public:
		float4 origin;  // Gotta have 1 in w coordinate
		float4 dir;     // Gotta be normalized and have 0 in w coordinate

		__host__ __device__ Ray(const float4& origin, const float4& dir)
			: origin(origin)
			, dir(dir)
		{
			this->origin.w = 1.0f;
			this->dir.w = 0.0f;
			//this->dir = normalize(this->dir);
		}

		inline __host__ __device__ float4 at(float t) const
		{
			return origin + t * dir;
		}
	};
}
