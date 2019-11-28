//
// Created by pecatoma on 28.11.2019.
//

#ifndef VISUALIZATION_OPENGL_GLFWMANAGER_H
#define VISUALIZATION_OPENGL_GLFWMANAGER_H

#include <GLFW/glfw3.h>
#include <iostream>
#include "camera.h"

class WindowCreator{
public:
	static GLFWwindow *createGLFWWindow(int _width, int _height, Camera _camera);
private:
	static int width, height;
	static bool firstMouse;
	static float lastX;
	static float lastY;
	static Camera camera;

	static void framebuffer_size_callback(GLFWwindow *window, int width, int height);

	static void mouse_callback(GLFWwindow *window, double xpos, double ypos);

	static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

};

#endif //VISUALIZATION_OPENGL_GLFWMANAGER_H
