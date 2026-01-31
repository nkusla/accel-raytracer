#pragma once
#include "IHittable.hpp"
#include "cuda_compat.hpp"

class World : public IHittable {
private:
	IHittable** objects;
	int num_objects;
	int capacity;
	color skyboxColor;

public:
	World() : objects(new IHittable*[8]), num_objects(0), capacity(8), skyboxColor(color(0.5f, 0.7f, 1.0f)) {}

	~World() {
		delete[] objects;
	}

	void add(IHittable* object) {
		if (num_objects >= capacity) {
			capacity = capacity * 2 + 1;
			IHittable** new_objects = new IHittable*[capacity];
			for (int i = 0; i < num_objects; i++) {
				new_objects[i] = objects[i];
			}
			delete[] objects;
			objects = new_objects;
		}
		objects[num_objects] = object;
		num_objects++;
	}

	__host__ __device__
	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const override;

	__host__ __device__
	color getSkyboxColor(const Ray& ray) const;
};