#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"
#include "World.hpp"
#include <iostream>
#include "Materials.hpp"
#include <chrono>
#include <cuda_runtime.h>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Initialization kernels - construct objects directly on device to set up proper vtables
__global__ void initMaterials(Lambertian* mat_ground, Lambertian* mat_center) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (mat_ground) Lambertian(color(0.8, 0.8, 0.0));
        new (mat_center) Lambertian(color(0.1, 0.2, 0.5));
    }
}

__global__ void initSpheres(Sphere* sphere_ground, Sphere* sphere_center,
                            Lambertian* mat_ground, Lambertian* mat_center) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (sphere_ground) Sphere(vec3(0.0, -100.5, -1.0), 100.0, mat_ground);
        new (sphere_center) Sphere(vec3(0.0, 0.0, -1.2), 0.5, mat_center);
    }
}

__global__ void initWorld(World* world, IHittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (world) World(objects, num_objects, num_objects);
    }
}

__global__ void initCamera(Camera* camera, float aspect_ratio, int image_width, int msaa_samples, int max_bounce) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (camera) Camera(aspect_ratio, image_width, msaa_samples, max_bounce);
    }
}

// Simple test kernel to verify virtual functions work
__global__ void testVirtualCall(Sphere* sphere, World* world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Ray test_ray(vec3(0, 0, 0), vec3(0, 0, -1));
        HitRecord rec;
        
        // Test direct sphere hit
        bool sphere_hit = sphere->hit(test_ray, 0.001f, 1000.0f, rec);
        printf("Direct sphere hit: %d\n", sphere_hit);
        
        // Test world hit
        bool world_hit = world->hit(test_ray, 0.001f, 1000.0f, rec);
        printf("World hit: %d\n", world_hit);
    }
}

int main() {
    // Setup parameters on host
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 16;  // Small image for testing
    int msaa_samples = 1;  // Reduce samples for faster testing
    int max_bounce = 8;
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // === Allocate materials on GPU ===
    Lambertian* d_material_ground;
    Lambertian* d_material_center;
    CUDA_CHECK(cudaMalloc(&d_material_ground, sizeof(Lambertian)));
    CUDA_CHECK(cudaMalloc(&d_material_center, sizeof(Lambertian)));

    // Construct materials ON DEVICE (not copy from host!)
    initMaterials<<<1, 1>>>(d_material_ground, d_material_center);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Materials initialized" << std::endl;

    // === Allocate spheres on GPU ===
    Sphere* d_sphere_ground;
    Sphere* d_sphere_center;
    CUDA_CHECK(cudaMalloc(&d_sphere_ground, sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&d_sphere_center, sizeof(Sphere)));

    // Construct spheres ON DEVICE (not copy from host!)
    initSpheres<<<1, 1>>>(d_sphere_ground, d_sphere_center, d_material_ground, d_material_center);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Spheres initialized" << std::endl;

    // === Create World on GPU ===
    // First, create the objects array on GPU
    IHittable** d_objects;
    CUDA_CHECK(cudaMalloc(&d_objects, 2 * sizeof(IHittable*)));

    // Copy sphere pointers to the objects array
    IHittable* h_objects[2] = { d_sphere_ground, d_sphere_center };
    CUDA_CHECK(cudaMemcpy(d_objects, h_objects, 2 * sizeof(IHittable*), cudaMemcpyHostToDevice));

    // Construct World ON DEVICE
    World* d_world;
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(World)));
    initWorld<<<1, 1>>>(d_world, d_objects, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "World initialized" << std::endl;

    // === Allocate and construct Camera on GPU ===
    Camera* d_camera;
    CUDA_CHECK(cudaMalloc(&d_camera, sizeof(Camera)));
    initCamera<<<1, 1>>>(d_camera, aspect_ratio, image_width, msaa_samples, max_bounce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Camera initialized" << std::endl;

    // === Allocate pixel buffer on GPU ===
    color* d_pixel_buffer;
    CUDA_CHECK(cudaMalloc(&d_pixel_buffer, image_width * image_height * sizeof(color)));

    // Increase device stack size for recursive ray tracing with virtual functions
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));  // Try 16KB
    std::cout << "Increased stack size for ray tracing" << std::endl;

    // === Launch kernel ===
    dim3 blockDim(2, 2);  // Start with just 4 threads
    dim3 gridDim(
        (image_width + blockDim.x - 1) / blockDim.x,
        (image_height + blockDim.y - 1) / blockDim.y
    );
    std::cout << "Launching render kernel with " << blockDim.x * blockDim.y * gridDim.x * gridDim.y << " threads..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    renderKernel<<<gridDim, blockDim>>>(d_camera, d_world, d_pixel_buffer, image_width, image_height, msaa_samples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Render time: " << elapsed.count() << " seconds" << std::endl;

    // === Copy result back to host ===
    color* h_pixel_buffer = new color[image_width * image_height];
    CUDA_CHECK(cudaMemcpy(h_pixel_buffer, d_pixel_buffer, image_width * image_height * sizeof(color), cudaMemcpyDeviceToHost));

    // === Write image ===
    PPMWriter::write_color(std::cout, h_pixel_buffer, image_width, image_height);
    std::cout << "Image saved to output.ppm" << std::endl;

    // === Cleanup ===
    delete[] h_pixel_buffer;
    CUDA_CHECK(cudaFree(d_pixel_buffer));
    CUDA_CHECK(cudaFree(d_camera));
    CUDA_CHECK(cudaFree(d_world));
    CUDA_CHECK(cudaFree(d_objects));
    CUDA_CHECK(cudaFree(d_sphere_ground));
    CUDA_CHECK(cudaFree(d_sphere_center));
    CUDA_CHECK(cudaFree(d_material_ground));
    CUDA_CHECK(cudaFree(d_material_center));

    return 0;
}
