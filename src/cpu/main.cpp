#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"
#include "World.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

color calc_ray_color(const Ray& ray, const World& world) {
	HitRecord hit_record;
	if (world.hit(ray, 0.001, INFINITY_F, hit_record)) {
		return 0.5f * (hit_record.normal + ONE);
	}

	return world.getSkyboxColor(ray);
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();

	Camera camera(16.0f / 9.0f, 800, 4);
	World world;

	world.add(std::make_shared<Sphere>(vec3(0.0f, 0.0f, -1.0f), 0.5f));
	world.add(std::make_shared<Sphere>(vec3(0.5f, 0.5f, -1.0f), 0.2f));

	int image_width = camera.getImageWidth();
	int image_height = camera.getImageHeight();
	auto pixel_buffer = new color[image_width * image_height];

	PPMWriter::write_header(std::cout, image_width, image_height);

	#ifdef _OPENMP
	std::clog << "OpenMP enabled, using " << omp_get_max_threads() << " threads\n" << std::flush;
	#endif
	#pragma omp parallel for collapse(2)
	for (int j = 0; j < image_height; j++) {
		#ifndef _OPENMP
		std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
		#endif
		for (int i = 0; i < image_width; i++) {
			color pixel_color(0.0f, 0.0f, 0.0f);

			for (int s = 0; s < camera.getMSAA(); s++) {
				Ray ray = camera.getRay(i, j, s);

				pixel_color += calc_ray_color(ray, world);
			}

			pixel_buffer[j * image_width + i] = pixel_color / float(camera.getMSAA());
		}
	}

	PPMWriter::write_color(std::cout, pixel_buffer, image_width, image_height);
	PPMWriter::write_footer(std::cout);
	delete[] pixel_buffer;

	std::clog << "\r\033[K" << "Rendering complete!" << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::clog << "Time taken: " << duration.count() << " ms" << std::endl;
}