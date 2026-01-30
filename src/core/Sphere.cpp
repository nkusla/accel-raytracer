#include "Sphere.hpp"

Sphere::Sphere(const vec3& center, float radius)
	: center(center), radius(radius) {}

bool Sphere::hit(const Ray& ray, double t_min, double t_max) const {
	vec3 oc = ray.origin - center;
	auto a = glm::dot(ray.direction, ray.direction);
	auto half_b = glm::dot(oc, ray.direction);
	auto c = glm::dot(oc, oc) - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0)
		return false;

	return true;
}