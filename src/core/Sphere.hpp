#include <glm/glm.hpp>
#include "IHittable.hpp"
#include "types.hpp"

class Sphere : public IHittable {
private:
	vec3 center;
	float radius;

public:
	Sphere(const vec3& center, float radius);

	bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hit_record) const override;
};
