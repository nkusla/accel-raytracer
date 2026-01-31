#pragma once
#include "IHittable.hpp"
#include "types.hpp"
#include "IMaterial.hpp"
#include "cuda_compat.hpp"

class Sphere : public IHittable {
private:
	vec3 center;
	float radius;
	IMaterial* material;
public:
	Sphere(const vec3& center, float radius, IMaterial* material);

	__host__ __device__
	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const override;
};
