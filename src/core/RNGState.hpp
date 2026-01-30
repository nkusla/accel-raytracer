#pragma once

struct RNGState {
	int x;
	int y;
	int sample;
	int bounce;

	RNGState(int x, int y, int s, int b = 0)
		: x(x), y(y), sample(s), bounce(b) {}

	RNGState next_bounce() const {
		return RNGState(x, y, sample, bounce + 1);
	}
};