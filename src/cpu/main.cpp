#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"

#include <iostream>

color ray_color(const Ray& ray, const IHittable& hittable) {
	HitRecord hit_record;
	if (hittable.hit(ray, 0.001, INFINITY_F, hit_record)) {
		return 0.5f * (hit_record.normal + ONE);
	}

	vec3 unit_direction = glm::normalize(ray.direction);
	float t = 0.5f * (unit_direction.y + 1.0f);
	return glm::mix(WHITE, color(0.5f, 0.7f, 1.0f), t);
}

int main() {
	Camera camera(16.0f / 9.0f, 800);
	Sphere sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);

	int image_width = camera.getImageWidth();
	int image_height = camera.getImageHeight();

	std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

	for (int j = 0; j < image_height; j++) {
		std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			Ray ray = camera.getRay(i, j);

			color pixel_color = ray_color(ray, sphere);
			PPMWriter::write_color(std::cout, pixel_color);
		}
	}

	std::clog << "\rDone.                 \n";
}