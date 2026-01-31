#pragma once
#include <memory>
#include "IHittable.hpp"
#include "types.hpp"
#include "IMaterial.hpp"

class Sphere : public IHittable {
private:
	vec3 center;
	float radius;
	std::shared_ptr<IMaterial> material;
public:
	Sphere(const vec3& center, float radius, std::shared_ptr<IMaterial> material);

	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const override;
};
