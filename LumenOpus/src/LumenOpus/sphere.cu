#include "LumenOpus/hittable/sphere.h"
#include "device_launch_parameters.h"

namespace LumenOpus
{
	Spheres::Spheres(uint64_t maxSize)
		: m_maxArraySize(maxSize)
		, m_dataSize(maxSize * ARRAY_COUNT)
		, ArraySize(0)
	{
		AllocateData();
	}

	Spheres::~Spheres()
	{
		FreeData();
	}

	__host__ __device__ void Spheres::Add(float X, float Y, float Z, float Radius)
	{
		// Casual return cause idk what happens on that gpu
		if (ArraySize == m_maxArraySize) return;

		float* ptr = &Data[ArraySize];
		*ptr = X;
		
		ptr += m_maxArraySize;
		*ptr = Y;

		ptr += m_maxArraySize;
		*ptr = Z;
		
		ptr += m_maxArraySize;
		*ptr = Radius;
		
		ArraySize++;
	}

	__host__ __device__ void Spheres::AllocateData()
	{
		Data = new float[m_dataSize];
	}

	__host__ __device__ void Spheres::FreeData()
	{
		delete[] Data;
		m_dataSize = m_maxArraySize = ArraySize = 0;
		Data = nullptr;
	}

	__host__ Spheres** Spheres::MakeItDevice(const Spheres& host)
	{
		Spheres** d_s;
		checkCudaErrors(cudaMalloc((void**)&d_s, sizeof(Spheres*)));

		CreateSpheresDevice<<<1, 1>>>(d_s, host.m_maxArraySize);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		CopyToDevice(d_s, host);

		return d_s;
	}
	__host__ void Spheres::DeleteDevice(Spheres** spheres)
	{
		DeleteSphereDevice<<<1, 1>>>(spheres);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(spheres));
	}

	__host__ void Spheres::CopyToDevice(Spheres** d_s, const Spheres& h_s)
	{
		int gridSize = 0, blockSize = 0;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, CopySphereToDevice, 0));

		float *d_host;

		size_t copySize = h_s.m_dataSize * sizeof(float);
		checkCudaErrors(cudaMalloc(&d_host, copySize));
		checkCudaErrors(cudaMemcpy(d_host, h_s.Data, copySize, cudaMemcpyDefault));

		CopySphereToDevice<<<gridSize, blockSize>>>(d_s, d_host, h_s.m_dataSize, h_s.ArraySize);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_host));
	}

	__global__ void CreateSpheresDevice(Spheres** s, size_t maxSize)
	{
		if (threadIdx.x | blockIdx.x) return;
		(*s) = new Spheres(maxSize);
#ifndef NDEBUG
		(*s)->m_type = DeviceType::GPU;
#endif
	}

	__global__ void DeleteSphereDevice(Spheres** s)
	{
		if (threadIdx.x | blockIdx.x) return;
		delete* (s);
	}

	__global__ void CopySphereToDevice(
		Spheres** s,
		const float* host,
		int N,
		uint64_t Size
		)
	{
		float* out = (*s)->Data;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
			out[i] = host[i];

		if (threadIdx.x | blockIdx.x) return;

		(*s)->ArraySize = Size;
	}
}
