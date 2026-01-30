#pragma once
#include <glm/glm.hpp>

using color = glm::vec3;
#define BLACK color(0.0f, 0.0f, 0.0f)
#define WHITE color(1.0f, 1.0f, 1.0f)

using vec3 = glm::vec3;
using vec2 = glm::vec2;
#define ZERO vec3(0.0f, 0.0f, 0.0f)
#define ONE vec3(1.0f, 1.0f, 1.0f)
#define UP vec3(0.0f, 1.0f, 0.0f)
#define DOWN vec3(0.0f, -1.0f, 0.0f)
#define LEFT vec3(-1.0f, 0.0f, 0.0f)
#define RIGHT vec3(1.0f, 0.0f, 0.0f)
#define FORWARD vec3(0.0f, 0.0f, 1.0f)

constexpr float INFINITY_F = std::numeric_limits<float>::infinity();
