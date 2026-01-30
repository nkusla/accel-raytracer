#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"
#include "World.hpp"
#include <iostream>
#include <memory>

color calc_ray_color(const Ray& ray, const World& world) {
	HitRecord hit_record;
	if (world.hit(ray, 0.001, INFINITY_F, hit_record)) {
		return 0.5f * (hit_record.normal + ONE);
	}

	return world.getSkyboxColor(ray);
}

int main() {
	Camera camera(16.0f / 9.0f, 800);
	World world;

	world.add(std::make_shared<Sphere>(vec3(0.0f, 0.0f, -1.0f), 0.5f));
	world.add(std::make_shared<Sphere>(vec3(0.5f, 0.5f, -1.0f), 0.2f));

	int image_width = camera.getImageWidth();
	int image_height = camera.getImageHeight();

	std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

	for (int j = 0; j < image_height; j++) {
		std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			Ray ray = camera.getRay(i, j);

			color pixel_color = calc_ray_color(ray, world);
			PPMWriter::write_color(std::cout, pixel_color);
		}
	}

	std::clog << "\rDone.                 \n";
}