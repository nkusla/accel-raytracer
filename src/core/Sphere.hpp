#include <glm/glm.hpp>
#include "IHittable.hpp"
#include "types.hpp"

class Sphere : public IHittable {
private:
	vec3 center;
	float radius;

public:
	Sphere(const vec3& center, float radius);

	bool hit(const Ray& ray, double t_min, double t_max) const override;
};
