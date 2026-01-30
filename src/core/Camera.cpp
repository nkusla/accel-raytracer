#include "Camera.hpp"
#include "hash.hpp"

Camera::Camera(float aspect_ratio, int image_width, int msaa_samples)
	: aspect_ratio(aspect_ratio),
	image_width(image_width),
	focal_length(1.0),
	msaa_samples(msaa_samples),
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
	return vec2(
		random_float(i, j, sample, 0) - 0.5f,
		random_float(i, j, sample, 1) - 0.5f
	);
}
