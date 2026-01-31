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
	Camera(float aspect_ratio, int image_width, int msaa_samples = 1, int max_bounce = 8);

	int getImageWidth() const { return image_width; }
	int getImageHeight() const { return image_height; }
	int getMSAA() const { return msaa_samples; }

	Ray getRay(int i, int j, int sample) const;
	vec2 getOffset(int i, int j, int sample) const;

	void render(const World& world, color* pixel_buffer) const;

private:
	void initialize();
	color traceRay(const Ray& ray, const World& world, RNGState& state) const;

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
};
