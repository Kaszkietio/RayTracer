#include "LumenOpus/light.h"
#include "device_launch_parameters.h"

LumenOpus::Lights::Lights(
	uint64_t maxSize,
	const float4& ambient
)
	: m_maxArraySize(maxSize)
	, m_dataSize(maxSize * ARRAY_COUNT)
	, ArraySize(0)
	, AmbientLight(ambient)
{
	AllocateData();
}

LumenOpus::Lights::~Lights()
{
	FreeData();
}

__host__ __device__ void LumenOpus::Lights::Add(
	const float4& diffuse, 
	const float4& specular, 
	float x, float y, float z)
{
	// Casual return cause idk what happens on that gpu
	if (ArraySize == m_maxArraySize) return;

	float* ptr = &Data[ArraySize];
	*ptr = diffuse.x;

	ptr += m_maxArraySize;
	*ptr = diffuse.y;

	ptr += m_maxArraySize;
	*ptr = diffuse.z;

	ptr += m_maxArraySize;
	*ptr = diffuse.w;

	ptr += m_maxArraySize;
	*ptr = specular.x;

	ptr += m_maxArraySize;
	*ptr = specular.y;

	ptr += m_maxArraySize;
	*ptr = specular.z;

	ptr += m_maxArraySize;
	*ptr = specular.w;

	ptr += m_maxArraySize;
	*ptr = x;

	ptr += m_maxArraySize;
	*ptr = y;

	ptr += m_maxArraySize;
	*ptr = z;

	ArraySize++;
}

__host__ __device__ void LumenOpus::Lights::AllocateData()
{
	Data = new float[m_dataSize];
}

__host__ __device__ void LumenOpus::Lights::FreeData()
{
	delete[] Data;
	m_dataSize = m_maxArraySize = ArraySize = 0;
	Data = nullptr;
}

__host__ LumenOpus::Lights** LumenOpus::Lights::MakeItDevice(const Lights& host)
{
	Lights** d;
	checkCudaErrors(cudaMalloc((void**)&d, sizeof(Lights**)));

	CreateLightsDevice<<<1, 1>>>(d, host.m_maxArraySize);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	CopyToDevice(d, host);

	return d;
}

__host__ void LumenOpus::Lights::DeleteDevice(Lights** lights)
{
	DeleteLightsDevice<<<1, 1>>>(lights);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(lights));
}

__host__ void LumenOpus::Lights::CopyToDevice(Lights** d, const Lights& h)
{
	int gridSize = 0, blockSize = 0;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, CopyLightsToDevice, 0));

	float* d_host;

	size_t copySize = h.m_dataSize * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_host, copySize));
	checkCudaErrors(cudaMemcpy(d_host, h.Data, copySize, cudaMemcpyDefault));

	CopyLightsToDevice<<<gridSize, blockSize>>>(d, d_host, h.m_dataSize, h.ArraySize);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_host));
}

__global__ void LumenOpus::CreateLightsDevice(Lights** d, size_t maxSize)
{
	if (threadIdx.x | blockIdx.x) return;
	(*d) = new Lights(maxSize);
#ifndef NDEBUG
	(*d)->m_type = DeviceType::GPU;
#endif
}

__global__ void LumenOpus::DeleteLightsDevice(Lights** d)
{
	if (threadIdx.x | blockIdx.x) return;
	delete* (d);
}

__global__ void LumenOpus::CopyLightsToDevice(
	Lights** d, 
	const float* host, 
	int N, 
	uint64_t Size)
{
	float* out = (*d)->Data;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
		out[i] = host[i];

	if (threadIdx.x | blockIdx.x) return;

	(*d)->ArraySize = Size;
}
