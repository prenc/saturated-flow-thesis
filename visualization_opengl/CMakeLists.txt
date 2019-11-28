cmake_minimum_required(VERSION 3.15)
project(visualization_opengl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(LIBRARIES_TO_LINK GL GLU glfw3 X11 Xxf86vm Xrandr pthread Xi dl Xinerama Xcursor)
set(SOURCE_FILES libs/glad.c utils/shader.h utils/camera.h libs/stb_image.cpp utils/textureLoader.h utils/glfwManager.cpp utils/glfwManager.h)

add_executable(visualization_opengl main.cpp ${SOURCE_FILES})
target_link_libraries(visualization_opengl ${LIBRARIES_TO_LINK})
