#include "LumenOpus/renderer.h"
#include "LumenOpus/init.h"
#include "LumenOpus/kernel.h"

#include "cuda_runtime.h"
#include "helper_math.h"

#include <iostream>
#include <random>

namespace LumenOpus
{
	Renderer::Renderer() : Renderer(0, 0, 
#ifdef NDEBUG
		//10000
		//5000
		//2500
		1000
		//500
		//50
		//20
#else
		//50
		20
		//10
		//5
		//1
#endif
		,
#ifdef NDEBUG
		//1
		//1000
		100
		//10
#else
		//1
		5
		//20
#endif
	)
	{}

	Renderer::Renderer(int32_t fbWidth, int32_t fbHeight, uint64_t spheresCount, uint64_t lightCount)
		: h_Spheres(spheresCount)
		, h_Lights(lightCount)
		, m_camera(fbWidth, fbHeight)
	{
		// Choose gpu in case of multi-gpu setup
		LumenOpus::on_setup();

		AllocateCamera();
		UpdateCamera(0.0f, 0.0f, 1.0f);

		AllocateFramebuffer(fbWidth, fbHeight);
		
		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_int_distribution<std::mt19937::result_type> dist(0, INT_MAX);
		std::uniform_real_distribution<> distf(0, 1000000);

		//h_Spheres.Add(
		//	0.0f, 0.0f, 0.0f, 0.5f,
		//	0.03f, 0.5f, 0.5f, 32.0f,
		//	1.0f, 0.0f, 0.0f
		//);
		//h_Spheres.Add(
		//	-5.0f, 0.0f, 0.0f, 0.5f,
		//	0.03f, 0.5f, 0.5f, 32.0f,
		//	1.0f, 0.0f, 0.0f
		//);
		//h_Spheres.Add(
		//	-10.0f, 0.0f, 0.0f, 0.5f,
		//	0.03f, 0.5f, 0.5f, 32.0f,
		//	1.0f, 0.0f, 0.0f
		//);

		//float4 tmpLight = make_float4(1.0f);

		//h_Lights.Add(
		//	1.0f * tmpLight,
		//	1.0f * tmpLight,
		//	0.0f, 2.0f, 1.0f
		//);
		//h_Lights.Add(
		//	1.0f * tmpLight,
		//	1.0f * tmpLight,
		//	-10.0f, 2.0f, 1.0f
		//);
		
		//h_Spheres.Add(0.0f, 0.0f, -1.0f, 0.5f);
		//h_Spheres.Add(0.0f, -10.0f, -1.0f, 9.0f);

		float tmp = max(20.0f, spheresCount / 10.0f);
		float tmp2 = -tmp;

		for (int i = 0; i < spheresCount; i++)
		{
			float x = distf(rng) / 1000000.0f;
			x = lerp(tmp2, tmp, x);

			float y = distf(rng) / 1000000.0f;
			y = lerp(tmp2, tmp, y);

			float z = distf(rng) / 1000000.0f;
			z = lerp(tmp2, tmp, z);

			float rad = distf(rng) / 1000000.0f;
			rad = lerp(2.5f, 7.5f, rad);
			//rad = lerp(0.1f, 2.0f, rad);

			float ka = distf(rng) / 1000000.0f;

			float kd = distf(rng) / 1000000.0f;
			
			float ks = distf(rng) / 1000000.0f;
			
			float shininess = distf(rng) / 1000000.0f;
			shininess = lerp(1.0f, 10.0f, shininess);
			shininess = roundf(shininess);
			shininess = 1 << (int)shininess;
			
			float r = distf(rng) / 1000000.0f;
			
			float g = distf(rng) / 1000000.0f;
			
			float b = distf(rng) / 1000000.0f;

			h_Spheres.Add(
				//0.0f, 0.0f, -1.0f, 0.5f,
				x, y, z, rad,
				ka, kd, ks, shininess,
				r, g, b);
		}

		for (uint64_t i = 0; i < lightCount; i++)
		{
			float sr = distf(rng) / 1000000.0f;
			
			float sg = distf(rng) / 1000000.0f;
			
			float sb = distf(rng) / 1000000.0f;
			
			float sa = distf(rng) / 1000000.0f;

			float dr = distf(rng) / 1000000.0f;
			
			float dg = distf(rng) / 1000000.0f;
			
			float db = distf(rng) / 1000000.0f;
			
			float da = distf(rng) / 1000000.0f;

			float x = distf(rng) / 1000000.0f;
			x = lerp(tmp2, tmp, x);

			float y = distf(rng) / 1000000.0f;
			y = lerp(tmp2, tmp, y);

			float z = distf(rng) / 1000000.0f;
			z = lerp(tmp2, tmp, z);

			h_Lights.Add(
				make_float4(dr, dg, db, da),
				make_float4(sr, sg, sb, sa),
				x, y, z
				);
		}

		// Allocate spheres and lights at gpu
		d_Spheres = Spheres::MakeItDevice(h_Spheres);
		d_Lights = Lights::MakeItDevice(h_Lights);
	}

	Renderer::~Renderer()
	{
		FreeCamera();

		// Free framebuffer
		FreeFramebuffer();

		h_Spheres.FreeData();
		Spheres::DeleteDevice(d_Spheres);
		h_Lights.FreeData();
		Lights::DeleteDevice(d_Lights);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		LumenOpus::on_exit();
	}

	void Renderer::OnUpdate(uint32_t* data, int32_t width, int32_t height)
	{
		int gridSize = 0, blockSize = 0;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, render_pixel, 0));

		// Copy current camera position
		checkCudaErrors(cudaMemcpy(
			d_cameraPosition, 
			&h_cameraPosition, 
			sizeof(float3), 
			cudaMemcpyHostToDevice));

		// Render buffer
		render_pixel<<<gridSize, blockSize>>> (
			md_fb, 
			d_Spheres,
			d_Lights,
			m_camera,
			(float4*)d_cameraPosition, 
			h_angleYAxis,
			m_fbWidth, 
			m_fbHeight);
		checkCudaErrors(cudaGetLastError());

		// Synchronize
		checkCudaErrors(cudaDeviceSynchronize());

		// Copy rendered buffer to cpu mem
		checkCudaErrors(cudaMemcpy(data, md_fb, GetSize(), cudaMemcpyDeviceToHost));
	}

	void Renderer::OnResize(int32_t width, int32_t height)
	{
		FreeFramebuffer();
		AllocateFramebuffer(width, height);
		m_camera.OnResize(width, height);
	}

	void Renderer::UpdateCamera(const float& x, const float& y, const float& z)
	{
		h_cameraPosition.x = x;
		h_cameraPosition.y = y;
		h_cameraPosition.z = z;
	}

	void Renderer::MoveCamera(const float& dx, const float& dy, const float& dz)
	{
		h_cameraPosition.x += dx;
		h_cameraPosition.y += dy;
		h_cameraPosition.z += dz;
	}

	void Renderer::MoveCamera(const float& forward, const float& up, const float& right, const float& yaw)
	{
		m_camera.OnUpdate(forward, up, right, yaw);
		return;


		constexpr float3 upDirection{ 0.0f, 1.0f, 0.0f };
		constexpr float piRatio = 3.14159265358979323846f / 180.0f;

		h_rightDirection = cross(h_forwardDirection, upDirection);

		h_cameraPosition += h_forwardDirection * forward * VELOCITY;
		h_cameraPosition += h_rightDirection * right * VELOCITY;
		h_cameraPosition += upDirection * up * VELOCITY;

		if (yaw == 0.0f) return;

		float newAngle = yaw * VELOCITY_ANGLE;
		h_angleYAxis += newAngle;
		if (h_angleYAxis >= 360.0f) h_angleYAxis -= 360.0f;
		else if (h_angleYAxis < 0.0f) h_angleYAxis += 360.0f;

		float radian = newAngle * piRatio;
		float sine = sinf(radian), cosine = cosf(radian);
		float newX = h_forwardDirection.x * cosine + h_forwardDirection.z * sine;
		float newZ = h_forwardDirection.z * cosine - h_forwardDirection.x * sine;

		float3 tmp = make_float3(newX, 0.0f, newZ);

		h_forwardDirection.x = tmp.x;
		h_forwardDirection.z = tmp.z;
	}

	void Renderer::AllocateCamera()
	{
		checkCudaErrors(cudaMalloc(&d_cameraPosition, sizeof(float4)));
	}

	void Renderer::FreeCamera()
	{
		checkCudaErrors(cudaFree(d_cameraPosition));
	}

	void Renderer::AllocateFramebuffer(int32_t width, int32_t height)
	{
		m_fbWidth = width; m_fbHeight = height;

		if (width == 0 && height == 0)
		{
			md_fb = nullptr;
			return;
		}

		size_t fb_size = width * height * sizeof(uint32_t);
		checkCudaErrors(cudaMalloc((void**)&md_fb, fb_size));
	}

	void Renderer::FreeFramebuffer()
	{
		if (!md_fb) return;

		checkCudaErrors(cudaFree(md_fb));
		md_fb = nullptr;
		m_fbWidth = m_fbHeight = 0;
	}
}
