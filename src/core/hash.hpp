#pragma once
#include <cstdint>
#include <glm/glm.hpp>

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