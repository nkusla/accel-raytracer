#include "cuda_compat.hpp"
#include "Camera.hpp"
#include "random.hpp"
#include <cstdio>

__host__ __device__
void Camera::initialize() {
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

__host__ __device__
Ray Camera::getRay(int i, int j, int sample) const {
	auto offset = getOffset(i, j, sample);

	auto pixel_center = viewport_origin + ((i + offset.x) * pixel_delta_u) + ((j + offset.y) * pixel_delta_v);
	auto ray_direction = pixel_center - camera_center;
	return Ray(camera_center, ray_direction);
}

__host__ __device__
vec2 Camera::getOffset(int i, int j, int sample) const {
	RNGState state(i, j, sample, 0);
	return vec2(
		random_float(state) - 0.5f,
		random_float(state.next_bounce()) - 0.5f
	);
}

__host__ __device__
color Camera::traceRay(const Ray& ray, const World* world, RNGState& state) const {
	color attenuation(1.0f, 1.0f, 1.0f);
	Ray current_ray = ray;
	HitRecord hit_record;

	for (int bounce = 0; bounce < max_bounce; bounce++) {
		if (world->hit(current_ray, 0.001f, INFINITY_F, hit_record)) {
			auto scatter_direction = hit_record.normal + random_on_hemisphere_with_normal(state, hit_record.normal);
			current_ray = Ray(hit_record.point, scatter_direction);
			attenuation *= hit_record.material->albedo;
		} else {
			return attenuation * world->getSkyboxColor(current_ray);
		}
	}
	return BLACK;
}

__host__
void Camera::render(const World& world, color* pixel_buffer) const {
#ifdef _OPENMP
	#pragma omp parallel for collapse(2)
#endif
	for (int j = 0; j < image_height; j++) {
		for (int i = 0; i < image_width; i++) {
			color pixel_color(0.0f, 0.0f, 0.0f);

			for (int s = 0; s < msaa_samples; s++) {
				Ray ray = getRay(i, j, s);
				RNGState state(i, j, s, 0);

				pixel_color += traceRay(ray, &world, state);
			}

			pixel_buffer[j * image_width + i] = pixel_color / float(msaa_samples);
		}
	}
}

#ifdef __CUDACC__
__global__ void renderKernel(Camera* camera, World* world, color* pixel_buffer, int image_width, int image_height, int msaa_samples) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= image_width || j >= image_height)
		return;

	int pixel_index = j * image_width + i;
	color pixel_color(0.0f);

	for (int sample = 0; sample < msaa_samples; sample++) {
		Ray ray = camera->getRay(i, j, sample);
		RNGState state(i, j, sample, 0);
		pixel_color += camera->traceRay(ray, world, state);
	}

	pixel_color /= static_cast<float>(msaa_samples);

	pixel_buffer[pixel_index] = pixel_color;
}
#endif