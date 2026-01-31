#pragma once
#include <cstdint>
#include "cuda_compat.hpp"
#include "types.hpp"
#include "RNGState.hpp"

__host__ __device__
__forceinline__ uint32_t hash(uint32_t x) {
	x ^= x >> 16;
	x *= 0x7feb352d;
	x ^= x >> 15;
	x *= 0x846ca68b;
	x ^= x >> 16;
	return x;
}

__host__ __device__
__forceinline__ float hash_to_float(uint32_t hash) {
	return (hash & 0xFFFFFF) / 16777216.0f; // [0, 1)
}

__host__ __device__
__forceinline__ float random_float(const RNGState& state) {
	uint32_t seed = hash(state.x) ^ hash(state.y * 1973) ^ hash(state.sample * 9277) ^ hash(state.bounce * 26699);
	return glm::clamp(hash_to_float(hash(seed)), 0.0f, 1.0f);
}

__host__ __device__
__forceinline__ vec3 random_vec3(const RNGState& state) {
	return vec3(random_float(state), random_float(state), random_float(state));
}

__host__ __device__
__forceinline__ vec3 random_on_hemisphere(RNGState& state) {
	float u = random_float(state);
	float v = random_float(state.next_bounce());
	float phi = 2.0f * PI * u;
	float cos_theta = v;
	float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);
	return vec3(glm::cos(phi) * sin_theta, glm::sin(phi) * sin_theta, cos_theta);
}

__host__ __device__
__forceinline__ vec3 random_on_hemisphere_with_normal(RNGState& state, const vec3& normal) {
	vec3 rand_unit_vec = glm::normalize(random_on_hemisphere(state));
	return glm::dot(rand_unit_vec, normal) > 0 ? rand_unit_vec : -rand_unit_vec;
}