file(GLOB CORE_SOURCES "*.cpp")
add_library(core ${CORE_SOURCES})

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(core PUBLIC glm::glm)
