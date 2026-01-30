#pragma once

#include <glm/glm.hpp>
#include <iostream>

class PPMWriter {
public:
	static void write_header(std::ostream& out, int width, int height) {
		out << "P3\n" << width << ' ' << height << "\n255\n";
	}

	static void write_color(std::ostream& out, const glm::vec3& pixel_color) {
		out << static_cast<int>(255.999 * pixel_color.r) << ' '
			<< static_cast<int>(255.999 * pixel_color.g) << ' '
			<< static_cast<int>(255.999 * pixel_color.b) << '\n';
	}

	static void write_footer(std::ostream& out) {
		out << std::flush;
	}
};