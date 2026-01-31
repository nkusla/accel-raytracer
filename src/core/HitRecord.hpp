#pragma once
#include "types.hpp"
#include "Ray.hpp"
#include "cuda_compat.hpp"


class Material {
public:
	color albedo;

	__host__ __device__
	Material(const color& albedo) : albedo(albedo) {}
};

class HitRecord {
public:
	vec3 point;
	vec3 normal;
	float t;
	bool front_face;
	Material* material;

	__host__ __device__
	void set_face_normal(const Ray& ray, const vec3& unit_normal) {
		front_face = glm::dot(ray.direction, unit_normal) < 0;
		normal = front_face ? unit_normal : -unit_normal;
	}
};
