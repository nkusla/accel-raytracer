find_package(OpenMP)

file(GLOB CPU_SOURCES "*.cpp")

# Sequential (uses core without OpenMP)
add_executable(raytracer-seq ${CPU_SOURCES})
target_link_libraries(raytracer-seq PRIVATE core)

# OpenMP (uses core-omp with OpenMP)
if(OpenMP_CXX_FOUND)
	add_executable(raytracer-omp ${CPU_SOURCES})
	target_link_libraries(raytracer-omp PRIVATE core-omp)
endif()
