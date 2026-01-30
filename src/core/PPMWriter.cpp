#include "PPMWriter.hpp"

void PPMWriter::write_header(std::ostream& out, int width, int height) {
	out << "P3\n" << width << ' ' << height << "\n255\n";
}

void PPMWriter::write_color(std::ostream& out, const color& pixel_color) {
	out << static_cast<int>(255.999 * pixel_color.r) << ' '
		<< static_cast<int>(255.999 * pixel_color.g) << ' '
		<< static_cast<int>(255.999 * pixel_color.b) << '\n';
}

void PPMWriter::write_color(std::ostream& out, const color* pixel_buffer, int width, int height) {
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			out << static_cast<int>(255.999 * pixel_buffer[j * width + i].r) << ' '
				<< static_cast<int>(255.999 * pixel_buffer[j * width + i].g) << ' '
				<< static_cast<int>(255.999 * pixel_buffer[j * width + i].b) << '\n';
		}
	}
}

void PPMWriter::write_footer(std::ostream& out) {
	out << std::flush;
}
