#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "types.hpp"

#include <iostream>

color ray_color(const Ray& ray) {
	vec3 unit_direction = glm::normalize(ray.direction);
	float t = 0.5f * (unit_direction.y + 1.0f);
	return glm::mix(WHITE, color(0.5f, 0.7f, 1.0f), t);
}

int main() {
	Camera camera(16.0f / 9.0f, 800);

	int image_width = camera.getImageWidth();
	int image_height = camera.getImageHeight();

	std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

	for (int j = 0; j < image_height; j++) {
		std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			Ray ray = camera.getRay(i, j);

			color pixel_color = ray_color(ray);
			PPMWriter::write_color(std::cout, pixel_color);
		}
	}

	std::clog << "\rDone.                 \n";
}