#pragma once
#include "Sphere.hpp"
#include "cuda_compat.hpp"
#include "HitRecord.hpp"

class World {
private:
	Sphere** spheres;
	int num_spheres;
	int capacity;
	color skyboxColor;

public:
	World() : spheres(new Sphere*[8]), num_spheres(0), capacity(8), skyboxColor(color(0.5f, 0.7f, 1.0f)) {}

	// GPU-friendly constructor: takes pre-allocated device array
	__host__ __device__
	World(Sphere** spheres_array, int num_spheres, int capacity)
		: spheres(spheres_array), num_spheres(num_spheres), capacity(capacity), skyboxColor(color(0.5f, 0.7f, 1.0f)) {}

	__host__ __device__
	~World() {
		#ifndef __CUDACC__
		if (capacity > 0 && num_spheres <= 8) {
			delete[] spheres;
		}
		#endif
	}

	__host__
	void add(Sphere* sphere);

	__host__ __device__
	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const;

	__host__ __device__
	color getSkyboxColor(const Ray& ray) const;
};
