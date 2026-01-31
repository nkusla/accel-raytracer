#pragma once
#include "cuda_compat.hpp"
#include "HitRecord.hpp"
#include "random.hpp"

enum MaterialType { LAMBERTIAN = 0, METAL };

class Lambertian {
private:
	color albedo;

public:
	__host__ __device__
	Lambertian(const color& albedo) : albedo(albedo) {}

	__host__ __device__
	bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const {
		auto scatter_direction = hit_record.normal + random_on_hemisphere_with_normal(state, hit_record.normal);
		scattered = Ray(hit_record.point, scatter_direction);
		attenuation = albedo;
		return true;
	}
};

class Metal {
private:
	color albedo;
	float fuzz;
public:
	__host__ __device__
	Metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

	__host__ __device__
	bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const {
		auto scatter_direction = reflect(glm::normalize(ray.direction), hit_record.normal);
		scattered = Ray(hit_record.point, scatter_direction + fuzz * random_vec3(state));
		attenuation = albedo;
		return glm::dot(scattered.direction, hit_record.normal) > 0;
	}
};

class Material {
public:
	MaterialType type;
	union {
		Lambertian lambert;
		Metal metal;
	} data;

	__host__ __device__
	Material() : type(LAMBERTIAN), data{Lambertian(color(0,0,0))} {}

	__host__ __device__
	static Material make_lambertian(const color& albedo) {
		Material m = Material();
		m.type = LAMBERTIAN;
		m.data.lambert = Lambertian(albedo);
		return m;
	}

	__host__ __device__
	static Material make_metal(const color& albedo, float fuzz) {
		Material m = Material();
		m.type = METAL;
		m.data.metal = Metal(albedo, fuzz);
		return m;
	}

	__host__ __device__
	bool scatter(const Ray& ray, const HitRecord& hit_record, color& attenuation, RNGState& state, Ray& scattered) const {
		switch (type) {
			case LAMBERTIAN: return data.lambert.scatter(ray, hit_record, attenuation, state, scattered);
			case METAL: return data.metal.scatter(ray, hit_record, attenuation, state, scattered);
		}
		return false;
	}
};
