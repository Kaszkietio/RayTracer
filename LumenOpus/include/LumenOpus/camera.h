#pragma once
#include "cuda_runtime.h"

#include "helper_math.h"
#include "LumenOpus/ray.h"
#include "LumenOpus/enums.h"

namespace LumenOpus
{
	__device_builtin__ __builtin_align__(16) 
	class Camera
	{
	public:
		static constexpr float PiRatio = 3.14159265358979323846f / 180.0f;

		static constexpr float RotationSpeed = 50.0f;
		static constexpr float Speed = 30.0f;
#ifndef NDEBUG
	private:
		DeviceType type{ DeviceType::CPU };
#endif
	public:
		float3 Position{};
		float ImageAspectRatio{};
		float3 ForwardDirection{};
		float ViewportWidth{};
		float ViewportHeight{};
		float Angle{};
		float Sine{};
		float Cosine{};
	public:
		__host__ __device__ Camera(
			float v_width, 
			float v_height, 
			float3 pos = {0.0f, 0.0f, 0.0f}, 
			float3 forward = {0.0f, 0.0f, -1.0f}
		)
			: ViewportWidth(v_width)
			, ViewportHeight(v_height)
			, Position(pos)
			, ForwardDirection(forward)
			, Angle(0.f)
			, Cosine(1.0f)
			, Sine(0.0f)
		{
			ImageAspectRatio = ViewportWidth / ViewportHeight;
		}

		__host__ __device__ ~Camera()
		{}

		__host__ void OnUpdate(
			const float& forward,
			const float& up,
			const float& right,
			const float& yaw
		)
		{
			constexpr float3 upDirection{ 0.0f, 1.0f, 0.0f };
			float3 rightDirection = cross(ForwardDirection, upDirection);

			Position += ForwardDirection * Speed * forward;
			Position += rightDirection * Speed * right;
			Position += upDirection * Speed * up;

			if(yaw == 0.0f) return;

			float newAngle = yaw * RotationSpeed;
			Angle += newAngle;
			if (Angle >= 360.0f) Angle -= 360.0f;
			else if (Angle < 0.0f) Angle += 360.0f;

			float radian = newAngle * PiRatio;
			float sine = sinf(radian), cosine = cosf(radian);
			float newX = ForwardDirection.x * cosine + ForwardDirection.z * sine;
			float newZ = ForwardDirection.z * cosine - ForwardDirection.x * sine;

			float3 tmp = make_float3(newX, 0.0f, newZ);

			ForwardDirection.x = tmp.x;
			ForwardDirection.z = tmp.z;

			Cosine = cosf(Angle * PiRatio);
			Sine = sinf(Angle * PiRatio);
		}

		inline __host__ __device__  float4 GetDirection(const float& x, const float& y)
		{
			float4 d = make_float4(
				(2.0f * x / ViewportWidth - 1.0f) * ImageAspectRatio,
				2.0f * y / ViewportHeight - 1.0f,
				-1.0f,
				0.0f
			);
			return make_float4(
				d.x * Cosine + d.z * Sine,
				d.y,
				d.z * Cosine - d.x * Sine,
				d.w
			);
		}

		inline __host__ void OnResize(float v_width, float v_height)
		{
			ViewportWidth = v_width; ViewportHeight = v_height; ImageAspectRatio = ViewportWidth / ViewportHeight;
		}

		//static __host__ Camera** MakeItDevice(const Camera& host);
		//static __host__ void DeleteDevice(Camera** dev);
	};
}
