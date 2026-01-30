#pragma once
#include "types.hpp"
#include <glm/glm.hpp>

class Ray {
public:
	vec3 origin;
	vec3 direction;

	Ray() : origin(vec3(0.0f)), direction(vec3(0.0f)) {}

	Ray(const vec3& origin, const vec3& direction) : origin(origin), direction(direction) {}

	vec3 at(float t) const {
		return origin + t * direction;
	}
};