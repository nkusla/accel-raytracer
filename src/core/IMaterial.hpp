#pragma once
#include "Ray.hpp"
#include "types.hpp"
#include "RNGState.hpp"

class HitRecord;

class IMaterial {
public:
	virtual ~IMaterial() = default;

	virtual bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const = 0;
};