#include "Ray.hpp"

class IHittable {
public:
	virtual ~IHittable() = default;

	virtual bool hit(const Ray& ray, double t_min, double t_max) const = 0;
};