#pragma once
#include "Ray.hpp"
#include <memory>
#include "IMaterial.hpp"

class HitRecord {
public:
	vec3 point;
	vec3 normal;
	float t;
	bool front_face;
	IMaterial* material;

	void set_face_normal(const Ray& ray, const vec3& unit_normal) {
		front_face = glm::dot(ray.direction, unit_normal) < 0;
		normal = front_face ? unit_normal : -unit_normal;
	}
};

class IHittable {
public:
	virtual ~IHittable() = default;

	virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const = 0;
};