#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include "types.hpp"

inline uint32_t hash(uint32_t x) {
	x ^= x >> 16;
	x *= 0x7feb352d;
	x ^= x >> 15;
	x *= 0x846ca68b;
	x ^= x >> 16;
	return x;
}

inline float hash_to_float(uint32_t hash) {
	return (hash & 0xFFFFFF) / 16777216.0f; // [0, 1)
}

inline float random_float(int x, int y, int sample, int bounce = 0) {
	uint32_t seed = hash(x) ^ hash(y * 1973) ^ hash(sample * 9277) ^ hash(bounce * 26699);
	return glm::clamp(hash_to_float(hash(seed)), 0.0f, 1.0f);
}

inline vec3 random_vec3(int x, int y, int sample, int bounce = 0) {
	return vec3(random_float(x, y, sample, bounce), random_float(x, y, sample, bounce), random_float(x, y, sample, bounce));
}

inline vec3 random_on_hemisphere(float u, float v) {
	float phi = 2.0f * PI * u;
	float cos_theta = v;
	float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);
	return vec3(glm::cos(phi) * sin_theta, glm::sin(phi) * sin_theta, cos_theta);
}