#pragma once
#include <cstdint>


namespace LumenOpus
{
	class Renderer
	{
		///////////////////////// F I E L D S //////////////////////////// 
	public:
		static const size_t CHANNEL_COUNT = 4;
		static const size_t FLOAT4_SIZE = 4 * sizeof(float);

		const float VELOCITY{ 0.750f };
		const float VELOCITY_ANGLE{ 100.0f };
	private:
		uint32_t* md_fb{ nullptr };
		int32_t m_fbWidth{ 0 };
		int32_t m_fbHeight{ 0 };

		void* d_cameraPosition;
		float h_angleYAxis{ 0.0f };
		uint8_t h_cameraPosition[FLOAT4_SIZE];
		uint8_t h_fordwardDirection[FLOAT4_SIZE];
		uint8_t h_rightDirection[FLOAT4_SIZE];
		
		///////////////// M E M B E R   F U N C T I O N S //////////////////////////// 
	public:
		Renderer();
		Renderer(int32_t fbWidth, int32_t fbHeight);
		~Renderer();

		uint32_t* GetFramebuffer() noexcept { return md_fb; }
		int32_t GetWidth() noexcept { return m_fbWidth; }
		int32_t GetHeight() noexcept { return m_fbHeight; }
		constexpr size_t GetSize() noexcept { return CHANNEL_COUNT * m_fbWidth * m_fbHeight; }

		/// <summary>
		/// Renders frame using gpu
		/// </summary>
		/// <param name="data">: Data with pixels</param>
		/// <param name="width">: Width of the screen in pixels</param>
		/// <param name="height">: Height of the screen in pixels</param>
		void OnUpdate(uint32_t* data, int32_t width, int32_t height);

		/// <summary>
		/// Called on framebuffer resize in order to allocate sufficient amount of memory
		/// </summary>
		/// <param name="width"></param>
		/// <param name="height"></param>
		void OnResize(int32_t width, int32_t height);

		void UpdateCamera(
			const float& x,
			const float& y,
			const float& z
			);

		void MoveCamera(
			const float& dx,
			const float& dy,
			const float& dz
		);

		void MoveCamera(
			const float& forward,
			const float& up,
			const float& right,
			const float& yaw
		);
	private:
		void AllocateCamera();
		void FreeCamera();

		/// <summary>
		/// Allocates Unified memory framebuffer for frame
		/// </summary>
		/// <param name="width"></param>
		/// <param name="height"></param>
		void AllocateFramebuffer(int32_t width, int32_t height);
		/// <summary>
		/// Frees allocated memory framebuffer
		/// </summary>
		void FreeFramebuffer();
	};
}
