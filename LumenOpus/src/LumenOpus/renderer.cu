#include "LumenOpus/renderer.h"
#include "LumenOpus/init.h"
#include "LumenOpus/kernel.h"

#include "cuda_runtime.h"
#include "helper_math.h"

#include <iostream>

namespace LumenOpus
{
	Renderer::Renderer() : Renderer(0, 0)
	{}

	Renderer::Renderer(int32_t fbWidth, int32_t fbHeight)
	{
		AllocateCamera();
		UpdateCamera(0.0f, 0.0f, 0.0f);

		AllocateFramebuffer(fbWidth, fbHeight);

		// Choose gpu in case of multi-gpu setup
		on_setup();
	}

	Renderer::~Renderer()
	{
		FreeCamera();

		// Free framebuffer
		FreeFramebuffer();

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		on_exit();
	}

	void Renderer::OnUpdate(uint32_t* data, int32_t width, int32_t height)
	{
		// Some constants
		constexpr int32_t tx = 8;
		constexpr int32_t ty = 8;

		dim3 blocks(width / tx + 1, height / ty + 1);
		dim3 threads(tx, ty);

		// Copy current camera position
		checkCudaErrors(cudaMemcpy(
			d_cameraPosition, 
			&h_cameraPosition, 
			sizeof(float4), 
			cudaMemcpyHostToDevice));

		// Render buffer
		render_pixel<<<blocks, threads>>> (
			md_fb, 
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
	}

	void Renderer::UpdateCamera(const float& x, const float& y, const float& z)
	{
		float4* tmp = (float4*)h_cameraPosition;
		tmp->x = x;
		tmp->y = y;
		tmp->z = z;
		tmp->w = 1.0f;
	}

	void Renderer::MoveCamera(const float& dx, const float& dy, const float& dz)
	{
		float4* tmp = (float4*)&(h_cameraPosition[0]);
		tmp->x += dx;
		tmp->y += dy;
		tmp->z += dz;
	}

	void Renderer::MoveCamera(const float& forward, const float& up, const float& right, const float& yaw)
	{
		throw;
		//constexpr float4 upDirection{ 0.0f, 1.0f, 0.0f, 0.0f };
		//constexpr float3 upDirection3{ 0.0f, 1.0f, 0.0f };
		//constexpr float piRatio = 3.14159265358979323846f / 180.0f;

		//float4* forward4 = (float4*)h_fordwardDirection;
		//float4* right4 = (float4*)h_rightDirection;
		//float4* pos = (float4*)h_cameraPosition;

		//*pos += *forward4 * forward * VELOCITY;
		//*pos += *right4 * right * VELOCITY;
		//*pos += upDirection * up * VELOCITY;

		//if (yaw == 0.0f) return;

		//h_angleYAxis += yaw * VELOCITY_ANGLE;
		//if (h_angleYAxis >= 360.0f) h_angleYAxis -= 360.0f;
		//else if (h_angleYAxis < 0.0f) h_angleYAxis += 360.0f;

		//float radian = h_angleYAxis * piRatio;
		//float sine = sinf(radian), cosine = cosf(radian);
		//float newX = forward4->x * cosine + forward4->z * sine;
		//float newZ = forward4->z * cosine - forward4->x * sine;

		//float3 tmp = normalize(make_float3(
		//	newX,
		//	0.0f,
		//	newZ
		//));

		//forward4->x = tmp.x;
		//forward4->z = tmp.z;
		//

		//float3 tmpR = cross(*(float3*)forward4, upDirection3);
		//right4->x = tmpR.x;
		//right4->y = tmpR.y;
		//right4->z = tmpR.z;
		//right4->w = 0.0f;
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
