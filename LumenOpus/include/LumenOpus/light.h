#pragma once
#include "cuda_runtime.h"
#include "helper_math.h"
#include "utils.h"

#include <cstdint>

#include "LumenOpus/enums.h"
#include "LumenOpus/error.h"

namespace LumenOpus
{
	// TODO
	__device_builtin__ __builtin_align__(16) 
	class Lights
	{
		// If changing (adding arrays) look at CopyToDevice function
		static constexpr uint64_t ARRAY_COUNT = 11;
		enum class Arrays : int32_t
		{
			DR, DG, DB, DA,
			SR, SG, SB, SA,
			 X,  Y,  Z, 
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
		float4 AmbientLight{};

		// Main data
		// Its size is ARRAY_COUNT * m_maxArraySize
		float* Data;
	public:
		__host__ __device__ Lights(
			uint64_t maxSize = 0,
			const float4& ambient = {1.0f, 1.0f, 1.0f, 1.0f}
		);
		__host__ __device__ ~Lights();

		__host__ __device__ void Add(
			const float4& diffuse,
			const float4& specular,
			float x, float y, float z
		);

		__host__ __device__ void AllocateData();
		__host__ __device__ void FreeData();

		constexpr __host__ __device__ float* GetArray(Arrays a)
		{
			return Data + m_maxArraySize * (int32_t)a;
		}

		__host__ __device__ float4 GetPosition(int index)
		{
			if (index >= ArraySize) return { 0 };

			float4 res{};
			float* ptr = &(GetArray(Arrays::X)[index]);
			res.x = *ptr;

			ptr += m_maxArraySize;
			res.y = *ptr;

			ptr += m_maxArraySize;
			res.z = *ptr;

			res.w = 1.0f;
			return res;
		}

		__host__ __device__ float4 GetAmbient()
		{
			return AmbientLight;
		}

		__host__ __device__ float4 GetDiffuse(int index)
		{
			if (index >= ArraySize) return { 0 };

			float4 res{};
			float* ptr = &(GetArray(Arrays::DR)[index]);
			res.x = *ptr;

			ptr += m_maxArraySize;
			res.y = *ptr;

			ptr += m_maxArraySize;
			res.z = *ptr;

			ptr += m_maxArraySize;
			res.w = *ptr;
			return res;
		}

		__host__ __device__ float4 GetSpecular(int index)
		{
			if (index >= ArraySize) return { 0 };

			float4 res{};
			float* ptr = &(GetArray(Arrays::SR)[index]);
			res.x = *ptr;

			ptr += m_maxArraySize;
			res.y = *ptr;

			ptr += m_maxArraySize;
			res.z = *ptr;

			ptr += m_maxArraySize;
			res.w = *ptr;
			return res;
		}

		////////////// STATIC MEMBER FUNCTIONS ////////////////////////
	public:
		static __host__ Lights** MakeItDevice(const Lights& host);
		static __host__ void DeleteDevice(Lights** lights);
	private:
		static __host__ void CopyToDevice(Lights** d, const Lights& h);
	};

	static __host__ __device__ float4 PhongModel(
		Lights* lights,
		const float4& mat,
		const float4& rayOrigin,
		const float4& spherePosition,
		const float4& hitPoint,
		const float4& objectColor
	)
	{
		// Attenuation coefficients
		constexpr float AC = 1.0f;
		//constexpr float AL = 0.09f;
		//constexpr float AQ = 0.032f;
		constexpr float AL = 0.01f;
		constexpr float AQ = 0.003f;
		float4 normal = normalize(hitPoint - spherePosition);
		float4 viewDir = normalize(rayOrigin - spherePosition);
		float4 reflectDir{};
		float4 lightPosition{}, lightDirection{}, ambientColor{}, diffuseColor{}, specularColor{};
		float4 distV{};
		float distSq, attuation;

		const float& ambientStrength = mat.x;
		const float& diffuseStrenght = mat.y;
		const float& specularStrength = mat.z;
		const float& shininess = mat.w;

		float4 ambient{}, diffuse{}, specular{};
		float4 sumDiffSpec = make_float4(0.0f);
		float spec;

		// Set ambient
		ambientColor = lights->GetAmbient();
		ambient = ambientStrength * ambientColor;

		// Work with diffuse and specular
		for (int i = 0; i < lights->ArraySize; i++)
		{
			lightPosition = lights->GetPosition(i);
			lightDirection = normalize(lightPosition - hitPoint);
			reflectDir = LumenOpus::reflect(-lightDirection, normal);
			distV = lightPosition - hitPoint;
			distSq = dot(distV, distV);
			//attuation = 1;
			attuation = AC + AL * sqrtf(distSq) + AQ * distSq;

			diffuseColor = lights->GetDiffuse(i);
			specularColor = lights->GetSpecular(i);

			diffuse = max((diffuseStrenght * dot(normal, lightDirection)), 0.0f) * diffuseColor;

			spec = dot(viewDir, reflectDir);
			spec = max(spec, 0.0f);
			spec = powf(spec, shininess);
			specular = specularStrength * spec * specularColor;

			sumDiffSpec += (diffuse + specular) / attuation;
		}

		float4 result = (ambient + sumDiffSpec) * objectColor;

		return clamp(result, 0.0f, 1.0f);
	}

	__global__ void CreateLightsDevice(Lights** d, size_t maxSize);

	__global__ void DeleteLightsDevice(Lights** d);

	__global__ void CopyLightsToDevice(
		Lights** d,
		const float* host,
		int N,
		uint64_t Size
		);
}
