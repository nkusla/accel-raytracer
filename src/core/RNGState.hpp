#pragma once
#include "cuda_compat.hpp"

struct RNGState {
	int x;
	int y;
	int sample;
	int bounce;

	__host__ __device__
	RNGState(int x, int y, int s, int b = 0)
		: x(x), y(y), sample(s), bounce(b) {}

	__host__ __device__
	RNGState& next_bounce() {
		bounce++;
		return *this;
	}
};