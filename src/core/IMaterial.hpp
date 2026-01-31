#pragma once
#include "Ray.hpp"
#include "types.hpp"
#include "RNGState.hpp"
#include "cuda_compat.hpp"

class HitRecord;

class IMaterial {
public:
	__host__ __device__
	virtual ~IMaterial() = default;

	__host__ __device__
	virtual bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const = 0;
};