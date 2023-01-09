#pragma once
#include "helper_math.h"
#include <cstdint>


namespace LumenOpus
{
	constexpr __host__ __device__ uint32_t to_rgba(
		const float& r,
		const float& g,
		const float& b,
		const float& a)
	{
        return
		  (uint8_t)(a * 255.99f) << 24		// A
        | (uint8_t)(b * 255.99f) << 16		// B
        | (uint8_t)(g * 255.99f) << 8		// G
        | (uint8_t)(r * 255.99f);			// R
	}

	inline __host__ __device__ uint32_t to_rgba(const float4& col)
	{
		return to_rgba(col.x, col.y, col.z, col.w);
	}

	inline __host__ __device__ uint32_t to_rgba(const float3& col)
	{
		return to_rgba(col.x, col.y, col.z, 1.0f);
	}

	/// <summary>
	/// Returns reflection of incident ray I around surface normal N. 
	/// I and N should have 0 in w coordinate.
	/// </summary>
	/// <param name="I">: )</param>
	/// <param name="N">: Should be normalized</param>
	/// <returns></returns>
	inline __host__ __device__ float4 reflect(const float4& I, const float4& N)
	{
		return I - 2.0f * N * dot(N, I);
	}
}

