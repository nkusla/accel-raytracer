#include "Sphere.hpp"

__host__ __device__
bool Sphere::hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const {
	vec3 oc = ray.origin - center;
	auto a = glm::dot(ray.direction, ray.direction);
	auto half_b = glm::dot(oc, ray.direction);
	auto c = glm::dot(oc, oc) - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0)
		return false;

	hit_record.t = (-half_b - glm::sqrt(discriminant)) / a;

	if (hit_record.t < t_min || hit_record.t > t_max)
		return false;

	hit_record.point = ray.at(hit_record.t);
	vec3 outward_normal = (hit_record.point - center) / radius;
	hit_record.set_face_normal(ray, outward_normal);
	hit_record.material = material;

	return true;
}