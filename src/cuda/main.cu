#include "Camera.hpp"
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
__global__ void initScene(Material* mat_ground, Material* mat_center,
                          Sphere* sphere_ground, Sphere* sphere_center,
                          Sphere** spheres, World* world, Camera* camera,
                          float aspect_ratio, int image_width, int msaa_samples, int max_bounce) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize materials
        new (mat_ground) Material(color(0.8, 0.8, 0.0));
        new (mat_center) Material(color(0.1, 0.2, 0.5));

        // Initialize spheres
        new (sphere_ground) Sphere(vec3(0.0, -100.5, -1.0), 100.0, mat_ground);
        new (sphere_center) Sphere(vec3(0.0, 0.0, -1.2), 0.5, mat_center);

        // Initialize sphere array
        spheres[0] = sphere_ground;
        spheres[1] = sphere_center;

        // Initialize world
        new (world) World(spheres, 2, 2);

        // Initialize camera
        new (camera) Camera(aspect_ratio, image_width, msaa_samples, max_bounce);
    }
}

int main() {
    // Setup parameters on host
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 800;
    int msaa_samples = 25;
    int max_bounce = 8;
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // === Allocate all scene objects on GPU ===
    Material* d_material_ground;
    Material* d_material_center;
    CUDA_CHECK(cudaMalloc(&d_material_ground, sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_material_center, sizeof(Material)));

    Sphere* d_sphere_ground;
    Sphere* d_sphere_center;
    CUDA_CHECK(cudaMalloc(&d_sphere_ground, sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&d_sphere_center, sizeof(Sphere)));

    Sphere** d_spheres;
    CUDA_CHECK(cudaMalloc(&d_spheres, 2 * sizeof(Sphere*)));

    World* d_world;
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(World)));

    Camera* d_camera;
    CUDA_CHECK(cudaMalloc(&d_camera, sizeof(Camera)));

    // === Initialize entire scene with single kernel launch ===
    initScene<<<1, 1>>>(d_material_ground, d_material_center,
                        d_sphere_ground, d_sphere_center,
                        d_spheres, d_world, d_camera,
                        aspect_ratio, image_width, msaa_samples, max_bounce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Allocate pixel buffer on GPU ===
    color* d_pixel_buffer;
    CUDA_CHECK(cudaMalloc(&d_pixel_buffer, image_width * image_height * sizeof(color)));

    // Increase device stack size for recursive ray tracing with virtual functions
    // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));  // Try 16KB

    // === Launch kernel ===
    dim3 blockDim(16, 16);
    dim3 gridDim(image_width / blockDim.x, image_height / blockDim.y);

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
    CUDA_CHECK(cudaFree(d_sphere_ground));
    CUDA_CHECK(cudaFree(d_sphere_center));
    CUDA_CHECK(cudaFree(d_material_ground));
    CUDA_CHECK(cudaFree(d_material_center));

    return 0;
}
