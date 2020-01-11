#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <iostream>

#include "utils/Window.h"
#include "utils/Chart.h"

void calculateFrameTime();
void initGlad();

int SCR_WIDTH = 1920;
int SCR_HEIGHT = 1080;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{
	Camera camera(glm::vec3(-2.5f, 0.7f, 0.0f));
	Window::createGLFWWindow(SCR_WIDTH, SCR_HEIGHT, &camera);
	initGlad();
	Chart chart = Chart("resources/heads_ca.txt", &camera, SCR_WIDTH, SCR_HEIGHT);

	while (Window::shouldClose())
	{
		calculateFrameTime();

		Window::processInput(deltaTime);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		chart.draw();

		Window::refreshWindow();
	}
	chart.delete_buffers();

	Window::terminateWindow();
	return 0;
}
void calculateFrameTime(){
	float currentFrame = glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;
}
void initGlad(){
	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit( -1);
	}
}