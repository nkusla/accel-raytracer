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


int main(int argc, char* argv[]) {
	int image_width = 800;
	int samples = 25;
	int max_bounce = 8;

	if (argc != 4) {
		std::clog << "Usage: " << argv[0] << " <image_width>" << " <samples>" << " <max_bounce>" << std::endl;
		std::clog << "Defaulting to image_width = 800" << std::endl;
		std::clog << "Defaulting to samples = 25" << std::endl;
		std::clog << "Defaulting to max_bounce = 8" << std::endl;
	} else {
		image_width = std::stoi(argv[1]);
		samples = std::stoi(argv[2]);
		max_bounce = std::stoi(argv[3]);
	}

	int image_height = int(image_width / (16.0f / 9.0f));

	Camera camera(16.0f / 9.0f, 800, samples, max_bounce);
	World world;

	auto material_ground = Material::make_lambertian(color(0.8, 0.8, 0.0));
	auto material_center = Material::make_lambertian(color(0.1, 0.2, 0.5));
	auto material_left = Material::make_metal(color(0.8, 0.8, 0.8), 0.0);
	auto material_right = Material::make_metal(color(0.8, 0.8, 0.8), 0.4);
	auto material_front = Material::make_metal(color(0.8, 0.0, 0.0), 0.0);

	world.add(new Sphere(vec3( 0.0, -100.5, -1.5), 100.0, material_ground));
	world.add(new Sphere(vec3( 0.0, 0.0, -1.7), 0.5, material_center));
	world.add(new Sphere(vec3(-1.0, 0.0, -1.0), 0.5, material_left));
	world.add(new Sphere(vec3( 1.0, 0.0, -1.0), 0.5, material_right));
	world.add(new Sphere(vec3( 0.0, -0.3, -1.0), 0.2, material_front));

	image_width = camera.getImageWidth();
	image_height = camera.getImageHeight();

	auto pixel_buffer = new color[image_width * image_height];

	PPMWriter::write_header(std::cout, image_width, image_height);

	#ifdef _OPENMP
	std::clog << "Running with OpenMP using " << omp_get_max_threads() << " threads\n" << std::flush;
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

	std::clog << "Image resolution: " << image_width << "x" << image_height << std::endl;
	std::clog << "Rendering time: " << duration.count() << " ms" << std::endl;
}