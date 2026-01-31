#pragma once
#include "IHittable.hpp"
#include "cuda_compat.hpp"

class World {  // Removed: public IHittable
private:
	IHittable** objects;
	int num_objects;
	int capacity;
	color skyboxColor;

public:
	World() : objects(new IHittable*[8]), num_objects(0), capacity(8), skyboxColor(color(0.5f, 0.7f, 1.0f)) {}

	// GPU-friendly constructor: takes pre-allocated device array
	__host__ __device__
	World(IHittable** objects_array, int num_objects, int capacity)
		: objects(objects_array), num_objects(num_objects), capacity(capacity), skyboxColor(color(0.5f, 0.7f, 1.0f)) {}

	// Destructor - only delete if on host and we allocated it
	~World() {
		// Don't delete if using pre-allocated device array
		// This is a host-only operation anyway
#ifndef __CUDA_ARCH__
		if (capacity > 0 && num_objects <= 8) {  // Only if we allocated it in default constructor
			delete[] objects;
		}
#endif
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
	inline bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const {
		bool hit_anything = false;
		float closest_t = t_max;

		for (int i = 0; i < num_objects; i++) {
			if (objects[i]->hit(ray, t_min, closest_t, hit_record)) {
				hit_anything = true;
				closest_t = hit_record.t;
			}
		}

		return hit_anything;
	}

	__host__ __device__
	inline color getSkyboxColor(const Ray& ray) const {
		vec3 unit_direction = glm::normalize(ray.direction);
		float t = 0.5f * (unit_direction.y + 1.0f);
		return glm::mix(WHITE, skyboxColor, t);
	}
};