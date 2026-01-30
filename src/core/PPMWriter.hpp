#pragma once
#include "types.hpp"
#include <ostream>

class PPMWriter {
public:
	static void write_header(std::ostream& out, int width, int height);
	static void write_color(std::ostream& out, const color& pixel_color);
	static void write_footer(std::ostream& out);
};