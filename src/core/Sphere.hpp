#pragma once
#include "types.hpp"
#include "Ray.hpp"
#include "cuda_compat.hpp"
#include "HitRecord.hpp"

class Sphere {
private:
	vec3 center;
	float radius;
	Material* material;
public:
	__host__ __device__
	Sphere(const vec3& center, float radius, Material* material)
		: center(center), radius(glm::max(0.0f, radius)), material(material) {}

	__host__ __device__
	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const;
};
