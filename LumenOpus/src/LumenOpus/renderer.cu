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
		2500
		//1000
		//50
		//20
#else
		//50
		20
		//10
		//5
		//1
#endif
	)
	{}

	Renderer::Renderer(int32_t fbWidth, int32_t fbHeight, uint64_t spheresCount)
		: h_Spheres(spheresCount)
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

		//h_Spheres.Add(0.0f, 0.0f, -1.0f, 0.5f);
		//h_Spheres.Add(0.0f, -10.0f, -1.0f, 9.0f);

		for (int i = 0; i < spheresCount; i++)
		{
			float x = distf(rng) / 1000000.0f;
			x = lerp(-100.0f, 100.0f, x);

			float y = distf(rng) / 1000000.0f;
			y = lerp(-100.0f, 100.0f, y);

			float z = distf(rng) / 1000000.0f;
			z = lerp(-110.0f, -10.0f, z);

			float r = distf(rng) / 1000000.0f;
			r = lerp(2.5f, 7.5f, r);

			h_Spheres.Add(x, y, z, r);
		}

		// Allocate spheres at gpu
		d_Spheres = Spheres::MakeItDevice(h_Spheres);
	}

	Renderer::~Renderer()
	{
		FreeCamera();

		// Free framebuffer
		FreeFramebuffer();

		h_Spheres.FreeData();
		Spheres::DeleteDevice(d_Spheres);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		LumenOpus::on_exit();
	}

	void Renderer::OnUpdate(uint32_t* data, int32_t width, int32_t height)
	{
		// Some constants
		constexpr int32_t tx = 8;
		constexpr int32_t ty = 8;

		dim3 blocks(width / tx + 1, height / ty + 1);
		dim3 threads(tx, ty);
		int gridSize = 0, blockSize = 0;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, render_pixel, 0));

		// Copy current camera position
		checkCudaErrors(cudaMemcpy(
			d_cameraPosition, 
			&h_cameraPosition, 
			sizeof(float3), 
			cudaMemcpyHostToDevice));

		// Render buffer
		//render_pixel<<<blocks, threads>>> (
		render_pixel<<<gridSize, blockSize>>> (
			md_fb, 
			d_Spheres,
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
