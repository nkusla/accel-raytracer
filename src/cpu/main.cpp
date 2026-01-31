#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"
#include "World.hpp"
#include "Materials.hpp"
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif


int main() {
	Camera camera(16.0f / 9.0f, 800, 25);
	World world;

	auto material_ground = Material::make_lambertian(color(0.8, 0.8, 0.0));
	auto material_center = Material::make_lambertian(color(0.1, 0.2, 0.5));

	world.add(new Sphere(vec3( 0.0, -100.5, -1.0), 100.0, &material_ground));
	world.add(new Sphere(vec3( 0.0, 0.0, -1.2), 0.5, &material_center));

	int image_width = camera.getImageWidth();
	int image_height = camera.getImageHeight();
	auto pixel_buffer = new color[image_width * image_height];

	PPMWriter::write_header(std::cout, image_width, image_height);

	#ifdef _OPENMP
	std::clog << "Running with OpenMP, using " << omp_get_max_threads() << " threads\n" << std::flush;
	#else
	std::clog << "Running sequentially\n" << std::flush;
	#endif

	auto start = std::chrono::high_resolution_clock::now();
	camera.render(world, pixel_buffer);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	PPMWriter::write_color(std::cout, pixel_buffer, image_width, image_height);
	PPMWriter::write_footer(std::cout);
	delete[] pixel_buffer;

	std::clog << "\r\033[K" << "Rendering complete!" << std::endl;
	std::clog << "Rendering time: " << duration.count() << " ms" << std::endl;
}