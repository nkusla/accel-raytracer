#pragma once
#include "types.hpp"
#include "Ray.hpp"

class Camera {
public:
	Camera(float aspect_ratio, int image_width, int msaa_samples = 1);

	int getImageWidth() const { return image_width; }
	int getImageHeight() const { return image_height; }
	int getMSAA() const { return msaa_samples; }

	Ray getRay(int i, int j, int sample) const;
	vec2 getOffset(int i, int j, int sample) const;

private:
	void initialize();

	int msaa_samples;
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
};
