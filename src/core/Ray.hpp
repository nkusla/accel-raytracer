#pragma once

#include <glm/glm.hpp>

class Ray {
private:
	glm::vec3 origin;
	glm::vec3 direction;

public:

	Ray() : origin(glm::vec3(0.0f)), direction(glm::vec3(0.0f)) {}

	Ray(const glm::vec3& origin, const glm::vec3& direction) : origin(origin), direction(direction) {}

	const glm::vec3& getOrigin() const { return origin; }

	const glm::vec3& getDirection() const { return direction; }

	glm::vec3 at(float t) const {
		return origin + t * direction;
	}
};