#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <iostream>

#include "utils/Window.h"
#include "utils/Chart.h"

void processInput(GLFWwindow *window);

int SCR_WIDTH = 800;
int SCR_HEIGHT = 600;

Camera camera(glm::vec3(-2.5f, 0.7f, 0.0f));

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{
	GLFWwindow *window = Window::createGLFWWindow(SCR_WIDTH, SCR_HEIGHT, &camera);
	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	Chart chart = Chart("heads_ca.txt", &camera, SCR_WIDTH, SCR_HEIGHT);

	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		chart.draw();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	chart.delete_buffers();
	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow *window)
{

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(FORWARD, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(LEFT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(RIGHT, deltaTime);
	}

}
