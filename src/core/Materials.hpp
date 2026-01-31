#pragma once
#include "IMaterial.hpp"
#include "Ray.hpp"
#include "types.hpp"
#include "random.hpp"
#include "IHittable.hpp"
#include "cuda_compat.hpp"

class Lambertian : public IMaterial {
private:
	color albedo;
public:
	__host__ __device__
	Lambertian(const color& albedo) : albedo(albedo) {}

	__host__ __device__
	bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const override {
		auto scatter_direction = hit_record.normal + random_on_hemisphere_with_normal(state, hit_record.normal);
		scattered = Ray(hit_record.point, scatter_direction);
		attenuation = albedo;
		return true;
	}
};

class Metal : public IMaterial {
private:
	color albedo;
	float fuzz;
public:
	__host__ __device__
	Metal(const color& albedo, float fuzz = 0.0f) : albedo(albedo), fuzz(fuzz) {}

	__host__ __device__
	bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const override {
		auto reflected = glm::normalize(glm::reflect(ray.direction, hit_record.normal));
		reflected += fuzz * random_on_hemisphere(state.next_bounce());
		scattered = Ray(hit_record.point, reflected);
		attenuation = albedo;
		return glm::dot(scattered.direction, hit_record.normal) > 0;
	}
};