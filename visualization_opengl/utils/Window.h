//
// Created by pecatoma on 28.11.2019.
//

#ifndef VISUALIZATION_OPENGL_WINDOW_H
#define VISUALIZATION_OPENGL_WINDOW_H

#include <GLFW/glfw3.h>
#include <iostream>
#include "camera.h"

class Window{
public:
	static GLFWwindow *createGLFWWindow(int _width, int _height, Camera *_camera);

	static bool shouldClose();

	static void processInput(float deltaTime);
	static void refreshWindow();

	static void terminateWindow();

private:
	static int width, height;
	static bool firstMouse;
	static float lastX;
	static float lastY;
	static Camera* camera;

	static void framebuffer_size_callback(GLFWwindow *window, int width, int height);

	static void mouse_callback(GLFWwindow *window, double xpos, double ypos);

	static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

	static void init_glfw();

	static void register_callback();

};

#endif //VISUALIZATION_OPENGL_WINDOW_H