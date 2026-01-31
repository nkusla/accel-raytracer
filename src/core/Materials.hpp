#pragma once
#include "IMaterial.hpp"
#include "Ray.hpp"
#include "types.hpp"
#include "random.hpp"
#include "IHittable.hpp"

class Lambertian : public IMaterial {
	private:
		color albedo;
	public:
		Lambertian(const color& albedo) : albedo(albedo) {}

		bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const override {
			auto scatter_direction = hit_record.normal + random_on_hemisphere_with_normal(state, hit_record.normal);
			scattered = Ray(hit_record.point, scatter_direction);
			attenuation = albedo;
			return true;
		}
	};