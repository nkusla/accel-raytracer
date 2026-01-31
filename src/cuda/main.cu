#include "Camera.hpp"
#include "PPMWriter.hpp"
#include "Sphere.hpp"
#include "types.hpp"
#include "World.hpp"
#include <iostream>
#include "Materials.hpp"
#include <chrono>

int main() {
    Camera camera(16.0f / 9.0f, 800, 25);
    World world;

    auto material_ground = new Lambertian(color(0.8, 0.8, 0.0));
    auto material_center = new Lambertian(color(0.1, 0.2, 0.5));
    auto material_left = new Metal(color(0.8, 0.8, 0.8));
    auto material_right = new Metal(color(0.8, 0.8, 0.8), 0.5f);

    world.add(new Sphere(vec3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(new Sphere(vec3( 0.0, 0.0, -1.2), 0.5, material_center));

    int image_width = camera.getImageWidth();
    int image_height = camera.getImageHeight();
    auto pixel_buffer = new color[image_width * image_height];

    delete[] pixel_buffer;
    delete material_ground;
    delete material_center;
    delete material_left;
    delete material_right;

    return 0;
}
