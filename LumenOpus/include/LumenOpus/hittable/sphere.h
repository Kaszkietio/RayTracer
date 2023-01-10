#pragma once
#include "cuda_runtime.h"

#include <cstdint>
#include "LumenOpus/enums.h"
#include "LumenOpus/hittable/hittable.h"
#include "LumenOpus/error.h"
#include <LumenOpus/ray.h>

namespace LumenOpus
{

	__device_builtin__ __builtin_align__(16) 
	class Spheres
	{
	public:
		// If changing (adding arrays) look at CopyToDevice function
		static const uint64_t ARRAY_COUNT = 4;
		enum class Arrays : int32_t
		{
			X = 0, Y, Z, RADIUS,
		};
	private:
		uint64_t m_dataSize{ 0 };
		uint64_t m_maxArraySize{ 0 };
#ifndef NDEBUG
	public:
		DeviceType m_type{ DeviceType::CPU };
#endif
	public:
		uint64_t ArraySize{ 0 };

		// Main data
		// Its size is ARRAY_COUNT * m_maxArraySize
		float* Data;

	public:
		__host__ __device__ Spheres(uint64_t maxSize = 0);
		__host__ __device__ ~Spheres();

		__host__ __device__ void Add(float x, float y, float z, float radius);
		constexpr __host__ __device__ float* GetArray(Arrays a)
		{
			return Data + m_maxArraySize * (int32_t)a;
		}

		__host__ __device__ void AllocateData();
		__host__ __device__ void FreeData();
		////////////// STATIC MEMBER FUNCTIONS ////////////////////////
	public:
		static __host__ Spheres** Spheres::MakeItDevice(const Spheres& host);
		static __host__ void DeleteDevice(Spheres** spheres);
	private:
		static __host__ void CopyToDevice(Spheres** d_s, const Spheres& h_s);
	};

	inline static __host__ __device__ int32_t single_hit(
		const float4& center,
		float& radius,
		const Ray& ray,
		const float& t_min, 
		const float& t_max, 
		HitRecord& rec) 
	{
		float4 origin = ray.origin - center;

		// (bx^2 + by^2 + bz^2)t^2 + 2(axbx + ayby + azbz)t + (ax^2 + ay^2 + az^2 - r^2)
		// a - ray origin
		// b - ray direction
		// r - radius
		// t - hit distance 
		float a = dot(ray.dir, ray.dir);
		float half_b = dot(origin, ray.dir);
		float c = dot(origin, origin) - radius * radius;

		float delta = half_b * half_b - a * c;

		if (delta < 0.0f) 
		{
			return 0;
		}

		float sqrtDelta = sqrtf(delta);

		// Select nearest t that lies in acceptable range
		float t = (-half_b - sqrtDelta) / a;
		if(t < t_min || t > t_max)
		{
			t = (-half_b + sqrtDelta) / a;
			if (t < t_min || t > t_max)
				return 0;
		}

		rec.t = t;
		rec.p = ray.at(rec.t);
		rec.normal = (rec.p - center) / radius;

		return 1;
	}

	inline static __device__ bool HitSpheres(
		Spheres* spheres,
		const Ray* ray,
		const float t_min,
		const float t_max,
		HitRecord* rec
	)
	{
		bool hitAnything = false;
		float curClosest = t_max;
		float4 curSphere{ 0.0f, 0.0f, 0.0f, 1.0f };
		float* x = spheres->GetArray(Spheres::Arrays::X);
		float* y = spheres->GetArray(Spheres::Arrays::Y);
		float* z = spheres->GetArray(Spheres::Arrays::Z);
		float* radius = spheres->GetArray(Spheres::Arrays::RADIUS);

		for (size_t i = 0; i < spheres->ArraySize; i++)
		{
			curSphere.x = x[i];
			curSphere.y = y[i];
			curSphere.z = z[i];

			if (single_hit(
				curSphere,
				radius[i],
				*ray,
				t_min,
				curClosest,
				*rec
			))
			{
				hitAnything = true;
				curClosest = rec->t;
			}
		}

		return hitAnything;
	}

	__global__ void CreateSpheresDevice(Spheres** s, size_t maxSize);

	__global__ void DeleteSphereDevice(Spheres** s);

	__global__ void CopySphereToDevice(
		Spheres** s,
		const float* host,
		int N,
		uint64_t Size
		);
}
