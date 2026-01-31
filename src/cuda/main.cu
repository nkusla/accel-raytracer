#include "Camera.hpp"
#include "Materials.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"
#include "World.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
					  << cudaGetErrorString(err) << std::endl; \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

// Unified initialization kernel - construct all objects in one kernel
__global__ void initScene(
						  Sphere** spheres, World* world, Camera* camera,
						  float aspect_ratio, int image_width, int msaa_samples, int max_bounce) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// Initialize materials
		Material mat_ground = Material::make_lambertian(color(0.8, 0.8, 0.0));
		Material mat_center = Material::make_lambertian(color(0.1, 0.2, 0.5));
		Material mat_left = Material::make_metal(color(0.8, 0.8, 0.8), 0.0);
		Material mat_right = Material::make_metal(color(0.8, 0.8, 0.8), 0.4);

		// Initialize spheres
		spheres[0] = new Sphere(vec3(0.0, -100.5, -1.0), 100.0, mat_ground);
		spheres[1] = new Sphere(vec3(0.0, 0.0, -1.2), 0.5, mat_center);
		spheres[2] = new Sphere(vec3(-1.0, 0.0, -1.0), 0.5, mat_left);
		spheres[3] = new Sphere(vec3( 1.0, 0.0, -1.0), 0.5, mat_right);

		new (world) World(spheres, 4, 4);

		new (camera) Camera(aspect_ratio, image_width, msaa_samples, max_bounce);
	}
}

int main() {
	float aspect_ratio = 16.0f / 9.0f;
	int image_width = 800;
	int msaa_samples = 25;
	int max_bounce = 8;
	int image_height = int(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1 : image_height;

	Sphere** d_spheres;
	CUDA_CHECK(cudaMalloc(&d_spheres, 4 * sizeof(Sphere*)));

	World* d_world;
	CUDA_CHECK(cudaMalloc(&d_world, sizeof(World)));

	Camera* d_camera;
	CUDA_CHECK(cudaMalloc(&d_camera, sizeof(Camera)));

	// === Initialize entire scene with single kernel launch ===
	initScene<<<1, 1>>>(
						d_spheres, d_world, d_camera,
						aspect_ratio, image_width, msaa_samples, max_bounce);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// === Allocate pixel buffer on GPU ===
	color* d_pixel_buffer;
	CUDA_CHECK(cudaMalloc(&d_pixel_buffer, image_width * image_height * sizeof(color)));

	// === Launch kernel ===
	dim3 blockDim(16, 16);
	dim3 gridDim(image_width / blockDim.x, image_height / blockDim.y);

	std::clog << "Launching CUDA kernel with " << gridDim.x << "x" << gridDim.y << " blocks and " << blockDim.x << "x" << blockDim.y << " threads" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	renderKernel<<<gridDim, blockDim>>>(d_camera, d_world, d_pixel_buffer, image_width, image_height, msaa_samples);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::clog << "Rendering time: " << duration.count() << " ms" << std::endl;

	// === Copy result back to host ===
	color* h_pixel_buffer = new color[image_width * image_height];
	CUDA_CHECK(cudaMemcpy(h_pixel_buffer, d_pixel_buffer, image_width * image_height * sizeof(color), cudaMemcpyDeviceToHost));

	PPMWriter::write_header(std::cout, image_width, image_height);
	PPMWriter::write_color(std::cout, h_pixel_buffer, image_width, image_height);
	PPMWriter::write_footer(std::cout);

	// === Cleanup ===
	delete[] h_pixel_buffer;
	CUDA_CHECK(cudaFree(d_pixel_buffer));
	CUDA_CHECK(cudaFree(d_camera));
	CUDA_CHECK(cudaFree(d_world));
	CUDA_CHECK(cudaFree(d_spheres));

	return 0;
}
