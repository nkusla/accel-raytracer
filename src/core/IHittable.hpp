#pragma once
#include "Ray.hpp"
#include "IMaterial.hpp"
#include "cuda_compat.hpp"

class HitRecord {
public:
	vec3 point;
	vec3 normal;
	float t;
	bool front_face;
	IMaterial* material;

	__host__ __device__
	void set_face_normal(const Ray& ray, const vec3& unit_normal) {
		front_face = glm::dot(ray.direction, unit_normal) < 0;
		normal = front_face ? unit_normal : -unit_normal;
	}
};

class IHittable {
public:
	virtual ~IHittable() = default;

	__host__ __device__
	virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const = 0;
};