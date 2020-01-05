//
// Created by pecatoma on 28.11.2019.
//
#include "Window.h"

int Window::width, Window::height;
bool Window::firstMouse;
float Window::lastX;
float Window::lastY;
Camera* Window::camera;
GLFWwindow *window;

GLFWwindow * Window::createGLFWWindow(int _width, int _height, Camera *_camera)
{
	width = _width;
	height = _height;
	camera = _camera;

	firstMouse = true;
	lastX = width / 2.0f;
	lastY = height / 2.0f;

	init_glfw();
	register_callback();

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	return window;
}
void Window::framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Window::mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera->ProcessMouseMovement(xoffset, yoffset);
}

void Window::scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
	camera->ProcessMouseScroll(yoffset);
}

void Window::init_glfw()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// glfw window creation
	// --------------------
	window = glfwCreateWindow(width, height, "LearnOpenGL", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}
	glfwMakeContextCurrent(window);
}

void Window::register_callback()
{
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
}

void Window::processInput(float deltaTime)
{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		{
			glfwSetWindowShouldClose(window, true);
		}

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		{
			camera->ProcessKeyboard(FORWARD, deltaTime);
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		{
			camera->ProcessKeyboard(BACKWARD, deltaTime);
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		{
			camera->ProcessKeyboard(LEFT, deltaTime);
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		{
			camera->ProcessKeyboard(RIGHT, deltaTime);
		}
}

bool Window::shouldClose(){
	return !glfwWindowShouldClose(window);
}

void Window::refreshWindow()
{
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void Window::terminateWindow()
{
	glfwTerminate();
}


