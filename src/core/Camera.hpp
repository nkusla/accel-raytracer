#pragma once
#include "types.hpp"
#include "Ray.hpp"
#include "World.hpp"
#include "RNGState.hpp"
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
	inline color traceRay(const Ray& ray, const World* world, RNGState& state) const {
		color attenuation(1.0f, 1.0f, 1.0f);
		Ray current_ray = ray;
		HitRecord hit_record;

		for (int bounce = 0; bounce < max_bounce; bounce++) {
			if (world->hit(current_ray, 0.001f, INFINITY_F, hit_record)) {
				Ray scattered;
				color scattered_attenuation;
				if(hit_record.material->scatter(current_ray, hit_record, scattered_attenuation, state.next_bounce(), scattered)) {
					current_ray = scattered;
					attenuation *= scattered_attenuation;
				}
				else {
					return attenuation * world->getSkyboxColor(current_ray);
				}
			} else {
				return attenuation * world->getSkyboxColor(current_ray);
			}
		}
		return BLACK;
	}

	__host__
	void render(const World& world, color* pixel_buffer) const;

private:
	__host__ __device__
	void initialize() {
		image_height = int(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		viewport_height = 2.0;
		viewport_width = viewport_height * (float(image_width) / float(image_height));

		viewport_u = vec3(viewport_width, 0, 0);
		viewport_v = vec3(0, -viewport_height, 0);

		pixel_delta_u = viewport_u / float(image_width);
		pixel_delta_v = viewport_v / float(image_height);

		auto viewport_upper_left = camera_center
								 - vec3(0.0f, 0.0f, focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;
		viewport_origin = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
	}

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

// CUDA kernel declaration (defined in Camera.cpp)
__global__ void renderKernel(Camera* camera, World* world, color* pixel_buffer, int image_width, int image_height, int msaa_samples);
