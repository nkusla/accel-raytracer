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


int main() {
	Camera camera(16.0f / 9.0f, 800, 4);
	World world;

	world.add(std::make_shared<Sphere>(vec3(0.0f, 0.0f, -1.0f), 0.5f));
	world.add(std::make_shared<Sphere>(vec3(0.5f, 0.5f, -1.0f), 0.2f));

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