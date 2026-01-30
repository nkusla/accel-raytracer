#pragma once
#include <vector>
#include <memory>
#include "IHittable.hpp"

class World : public IHittable {
private:
	std::vector<std::shared_ptr<IHittable>> objects;
	color skyboxColor;

public:
	World() : skyboxColor(color(0.5f, 0.7f, 1.0f)) {}

	void add(std::shared_ptr<IHittable> object) {
		objects.push_back(object);
	}

	const std::vector<std::shared_ptr<IHittable>>& getObjects() const {
		return objects;
	}

	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const override;

	color getSkyboxColor(const Ray& ray) const;
};