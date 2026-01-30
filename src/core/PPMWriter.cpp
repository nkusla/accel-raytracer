#include "PPMWriter.hpp"

void PPMWriter::write_header(std::ostream& out, int width, int height) {
	out << "P3\n" << width << ' ' << height << "\n255\n";
}

void PPMWriter::write_color(std::ostream& out, const color& pixel_color) {
	out << static_cast<int>(255.999 * pixel_color.r) << ' '
		<< static_cast<int>(255.999 * pixel_color.g) << ' '
		<< static_cast<int>(255.999 * pixel_color.b) << '\n';
}

void PPMWriter::write_footer(std::ostream& out) {
	out << std::flush;
}
