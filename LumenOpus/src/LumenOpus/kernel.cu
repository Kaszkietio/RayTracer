﻿#include "LumenOpus/kernel.h"
#include "LumenOpus/utils.h"

#include "device_launch_parameters.h"
#include "helper_math.h"

#include <stdio.h>
#include <cstdint>

__global__ void LumenOpus::render_pixel(
    uint32_t* data, 
    float4* d_rayOrigin, 
    float angleYAxis,
    int32_t max_x, 
    int32_t max_y)
{
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Return if unnecessary thread
    if (x >= max_x || y >= max_y) return;

    constexpr float piRatio = 3.14159265358979323846f / 180.0f;
    int32_t index = max_x * y + x;
    float imageAspectRatio = float(max_x) / float(max_y);
    float4 rayOrigin = *d_rayOrigin;

    // Rotation data
    float radians = piRatio * angleYAxis;
    float cosine = cosf(radians);
    float sine = sinf(radians);

    // Fun time
    float4 spherePosition = make_float4(0.0f, 0.0f, -1.0f, 1.0f);
    float4 lightPosition = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    float sphereRadius = 0.5f;
    float3 sphereColor = make_float3(1.0f, 0.0f, 1.0f);
    uint32_t backColor = to_rgba(0.0f, 0.0f, 0.0f, 1.0f);
    float closestT;

    float4 rayDirection = make_float4(
		(2.0f * float(x) / float(max_x) - 1.0f) * imageAspectRatio,
        2.0f * float(y) / float(max_y) - 1.0f,
        -1.0f,
        0.0f
    );

    // Apply rotation
    rayDirection = make_float4(
        rayDirection.x * cosine + rayDirection.z * sine,
        rayDirection.y,
        rayDirection.z * cosine - rayDirection.x * sine,
        rayDirection.w
    );


    bool isHit = is_sphere_hit(
        rayOrigin,
        rayDirection,
        spherePosition,
        sphereRadius,
        closestT);

    if (!isHit || closestT < 0)
    {
        data[index] = backColor;
        return;
    }

    float4 hitPoint = rayOrigin + (closestT * rayDirection);
    hitPoint.w = 1.0f;
    
    float4 mat = make_float4(
        0.1f, //KA
        1.0f, //KD
        1.0f, //KS
        32.0f //Shininess
    );
    float4 lightColor = make_float4(1.0f);
    float4 objectColor = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	float4 result = PhongModel(
        mat,
        rayOrigin,
        spherePosition,
        lightPosition,
        lightColor,
        hitPoint,
        objectColor
    );

    data[index] = to_rgba(result);
}

__host__ __device__ LumenOpus::SphereHit LumenOpus::is_sphere_hit(const float3& rayOrigin, const float3& spherePosition, const float& sphereRadius)
{
    return {};
}

__host__ __device__ bool LumenOpus::is_sphere_hit(
    const float4& rayOrigin, 
	const float4& rayDirection,
	const float4& spherePosition, 
    const float& sphereRadius, 
    float& closestT)
{
    float4 origin = rayOrigin - spherePosition;

    // (bx^2 + by^2 + bz^2)t^2 + 2(axbx + ayby + azbz)t + (ax^2 + ay^2 + az^2 - r^2)
    // a - ray origin
    // b - ray direction
    // r - radius
    // t - hit distance 
    float a = dot(rayDirection, rayDirection);
    float b = 2.0f * dot(origin, rayDirection);
    float c = dot(origin, origin) - sphereRadius * sphereRadius;

    float delta = b * b - 4.0f * a * c;

    // return background color if no hit
    if (delta < 0) return false;

	closestT = (-b - sqrtf(delta)) / (2.0f * a);
    return true;
}

__host__ __device__ float4 LumenOpus::PhongModel(
    const float4& mat,
    const float4& rayOrigin,        // has to have 1 in w coord
    const float4& spherePosition,   // has to have 1 in w coord
    const float4& lightPosition,    // has to have 1 in w coord
    const float4& lightColor,
    const float4& hitPoint,         // has to have 1 in w coord
    const float4& objectColor
)
{
    float4 normal = normalize(hitPoint - spherePosition);
    float4 viewDir = normalize(rayOrigin - spherePosition);
    float4 lightDirection = normalize(lightPosition - hitPoint);
    float4 reflectDir = LumenOpus::reflect(-lightDirection, normal);

    const float& ambientStrength = mat.x;
    const float& diffuseStrenght = mat.y;
    const float& specularStrength = mat.z;
    const float& shininess = mat.w;

    float4 ambient = ambientStrength * lightColor;
    float4 diffuse = max((diffuseStrenght * dot(normal, lightDirection)), 0.0f) * lightColor;

    float spec = dot(viewDir, reflectDir);
    spec = max(spec, 0.0f);
    spec = powf(spec, shininess);
    float4 specular = specularStrength * spec * lightColor;

    float4 result = (ambient + diffuse + specular) * objectColor;

    return clamp(result, 0.0f, 1.0f);
}

__host__ __device__ float3 LumenOpus::PhongModel(
    const float4& mat, 
    const float3& rayOrigin, 
    const float3& spherePosition, 
    const float3& lightPosition, 
    const float3& lightColor, 
    const float3& hitPoint, 
    const float3& objectColor)
{
    float3 normal = normalize(hitPoint - spherePosition);
    float3 viewDir = normalize(rayOrigin - spherePosition);
    float3 lightDirection = normalize(lightPosition - hitPoint);
    float3 reflectDir = reflect(-lightDirection, normal);

    const float& ambientStrength = mat.x;
    const float& diffuseStrenght = mat.y;
    const float& specularStrength = mat.z;
    const float& shininess = mat.w;

    float3 ambient = ambientStrength * lightColor;
    float3 diffuse = max((diffuseStrenght * dot(normal, lightDirection)), 0.0f) * lightColor;

    float spec = dot(viewDir, reflectDir);
    spec = max(spec, 0.0f);
    spec = powf(spec, shininess);
    float3 specular = specularStrength * spec * lightColor;

    float3 result = (ambient + diffuse + specular) * objectColor;

    return clamp(result, 0.0f, 1.0f);
}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

int main2()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}