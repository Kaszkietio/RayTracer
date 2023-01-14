#pragma once
#include "cuda_runtime.h"
#include <cstdint>
#include "hittable/sphere.h"
#include "light.h"
#include "camera.h"

int main2();

namespace LumenOpus
{
	__global__ void render_pixel(
		uint32_t* data,
		Spheres** spheres,
		Lights** lightsptr,
		Camera camera,
		float4* d_rayOrigin,
		float angleYAxis,
		int32_t max_x,
		int32_t max_y);

	struct SphereHit
	{
		int32_t isHit;
		float closestT;
	};

	__host__ __device__ SphereHit is_sphere_hit(
		const float3& rayOrigin,
		const float3& spherePosition,
		const float& sphereRadius
	);

	__host__ __device__ bool is_sphere_hit(
		const float4& rayOrigin,
		const float4& rayDirection,
		const float4& spherePosition,
		const float& sphereRadius,
		float& closestT
	);

	__host__ __device__ float4 PhongModel(
		const float4& mat,
		const float4& rayOrigin,
		const float4& spherePosition,
		const float4& lightPosition,
		const float4& lightColor,
		const float4& hitPoint,
		const float4& objectColor
		);
	__host__ __device__ float3 PhongModel(
		const float4& mat,
		const float3& rayOrigin,
		const float3& spherePosition,
		const float3& lightPosition,
		const float3& lightColor,
		const float3& hitPoint,
		const float3& objectColor
		);
}
