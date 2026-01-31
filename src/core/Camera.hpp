#pragma once
#include "types.hpp"
#include "Ray.hpp"
#include "World.hpp"
#include "RNGState.hpp"
#include "random.hpp"
#include "HitRecord.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

class Camera {
public:
	__host__ __device__
	Camera(float aspect_ratio, int image_width, int msaa_samples = 1, int max_bounce = 8)
		: aspect_ratio(aspect_ratio),
		  image_width(image_width),
		  focal_length(1.0),
		  msaa_samples(msaa_samples),
		  max_bounce(max_bounce),
		  camera_center(ZERO)
	{
		initialize();
	}

	__host__ __device__
	int getImageWidth() const { return image_width; }
	__host__ __device__
	int getImageHeight() const { return image_height; }
	__host__ __device__
	int getMSAA() const { return msaa_samples; }

	__host__ __device__
	Ray getRay(int i, int j, int sample) const;
	__host__ __device__
	vec2 getOffset(int i, int j, int sample) const;

	__host__ __device__
	color traceRay(const Ray& ray, const World* world, RNGState& state) const;

	__host__
	void render(const World& world, color* pixel_buffer) const;

private:
	__host__ __device__
	void initialize();

	int msaa_samples;
	int max_bounce;
	int image_width;
	int image_height;
	float aspect_ratio;

	double focal_length;
	double viewport_height;
	double viewport_width;
	vec3 camera_center;

	vec3 viewport_u;
	vec3 viewport_v;

	vec3 pixel_delta_u;
	vec3 pixel_delta_v;

	vec3 viewport_origin;

	#ifdef __CUDACC__
	friend __global__ void renderKernel(Camera* camera, World* world, color* pixel_buffer, int image_width, int image_height, int msaa_samples);
	#endif
};

__global__ void renderKernel(Camera* camera, World* world, color* pixel_buffer, int image_width, int image_height, int msaa_samples);
