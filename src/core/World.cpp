#include "World.hpp"

bool World::hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const {
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

color World::getSkyboxColor(const Ray& ray) const {
	vec3 unit_direction = glm::normalize(ray.direction);
	float t = 0.5f * (unit_direction.y + 1.0f);
	return glm::mix(WHITE, skyboxColor, t);
}