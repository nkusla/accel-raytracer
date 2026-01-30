find_package(OpenMP)

file(GLOB CORE_SOURCES "*.cpp")

# Core library without OpenMP (for sequential builds)
add_library(core ${CORE_SOURCES})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PUBLIC glm::glm)

# Core library with OpenMP (for parallel builds)
if(OpenMP_CXX_FOUND)
	add_library(core-omp ${CORE_SOURCES})
	target_include_directories(core-omp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(core-omp PUBLIC glm::glm OpenMP::OpenMP_CXX)
endif()
