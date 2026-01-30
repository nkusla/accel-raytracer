#include "PPMWriter.hpp"
#include <glm/glm.hpp>

int main() {
    int image_width = 512;
    int image_height = 512;

    PPMWriter::write_header(std::cout, image_width, image_height);
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_color = glm::vec3(float(i)/(image_width-1), float(j)/(image_height-1), 0);
            PPMWriter::write_color(std::cout, pixel_color);
        }
    }
    PPMWriter::write_footer(std::cout);

    std::clog << "\rDone.                 \n";
    return 0;
}
