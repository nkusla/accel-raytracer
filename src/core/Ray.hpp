#pragma once
#include "types.hpp"
#include "cuda_compat.hpp"

class Ray {
public:
	vec3 origin;
	vec3 direction;

	__host__ __device__
	Ray() : origin(vec3(0.0f)), direction(vec3(0.0f)) {}

	__host__ __device__
	Ray(const vec3& origin, const vec3& direction) : origin(origin), direction(direction) {}

	__host__ __device__
	vec3 at(float t) const {
		return origin + t * direction;
	}
};