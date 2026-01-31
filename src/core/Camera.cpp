#include "Camera.hpp"
#include "random.hpp"

Camera::Camera(float aspect_ratio, int image_width, int msaa_samples, int max_bounce)
	: aspect_ratio(aspect_ratio),
	image_width(image_width),
	focal_length(1.0),
	msaa_samples(msaa_samples),
	max_bounce(max_bounce),
	camera_center(ZERO)
{
	initialize();
}

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

Ray Camera::getRay(int i, int j, int sample) const {
	auto offset = getOffset(i, j, sample);

	auto pixel_center = viewport_origin + ((i + offset.x) * pixel_delta_u) + ((j + offset.y) * pixel_delta_v);
	auto ray_direction = pixel_center - camera_center;
	return Ray(camera_center, ray_direction);
}

vec2 Camera::getOffset(int i, int j, int sample) const {
	RNGState state(i, j, sample, 0);
	return vec2(
		random_float(state) - 0.5f,
		random_float(state.next_bounce()) - 0.5f
	);
}

color Camera::traceRay(const Ray& ray, const World& world, RNGState& state) const {
	color accumulated_color(0.0f, 0.0f, 0.0f);
	color attenuation(1.0f, 1.0f, 1.0f);
	Ray current_ray = ray;
	HitRecord hit_record;

	for (int bounce = 0; bounce < max_bounce; bounce++) {
		if (world.hit(current_ray, 0.001f, INFINITY_F, hit_record)) {
			Ray scattered;
			color scattered_attenuation;
			if(hit_record.material->scatter(current_ray, hit_record, scattered_attenuation, state.next_bounce(), scattered)) {
				current_ray = scattered;
				attenuation *= scattered_attenuation;
			}
			else {
				return attenuation * world.getSkyboxColor(current_ray);
			}
		} else {
			return attenuation * world.getSkyboxColor(current_ray);
		}
	}
	return BLACK;
}

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

				pixel_color += traceRay(ray, world, state);
			}

			pixel_buffer[j * image_width + i] = pixel_color / float(msaa_samples);
		}
	}
}
