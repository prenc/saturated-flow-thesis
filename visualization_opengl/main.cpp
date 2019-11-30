#include <iostream>
#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include "libs/stb_image.h"
#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include "utils/textureLoader.h"
#include "utils/shader.h"
#include "utils/camera.h"
#include "utils/glfwManager.h"
#include "params.h"

void processInput(GLFWwindow *window);

void initDataBuffers(unsigned int *VAO, unsigned int *VBO);

void initCubePosition();
void read_file();

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
glm::vec3 cubePositions[ROWS][COLS];

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float graphData[ROWS][COLS];

// timing
float deltaTime = 0.0f;    // time between current frame and last frame
float lastFrame = 0.0f;

int main()
{
	read_file();
	GLFWwindow *window = WindowCreator::createGLFWWindow(SCR_WIDTH, SCR_HEIGHT, &camera);

	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glEnable(GL_DEPTH_TEST);

	Shader ourShader("vertex.glsl", "fragment.glsl");

	initCubePosition();

	unsigned int VBO, VAO;

	initDataBuffers(&VAO, &VBO);

	unsigned int texture1, texture2;
	std::string texture1Path = "resources/textures/awesomeface.png";
	std::string texture2Path = "resources/textures/container.jpg";
	loadTexture(&texture1, texture1Path.c_str(), texture1Path.size() + 1);
	loadTexture(&texture2, texture2Path.c_str(), texture2Path.size() + 1);

	ourShader.use();
	ourShader.setInt("texture1", 0);
	ourShader.setInt("texture2", 1);

	while (!glfwWindowShouldClose(window))
	{
		// per-frame time logic
		// --------------------
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		// -----
		processInput(window);

		// render
		// ------
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// bind textures on corresponding texture units
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texture2);

		ourShader.use();

		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f,
		                                        100.0f);
		ourShader.setMat4("projection", projection);

		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("view", view);

		glBindVertexArray(VAO);
		for (unsigned int i = 0; i < ROWS; i++)
		{
			for (int j = 0; j < COLS; ++j)
			{
				glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
				model = glm::translate(model, cubePositions[i][j]);
				ourShader.setMat4("model", model);

				glDrawArrays(GL_TRIANGLES, 0, 6);
			}
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);

	glfwTerminate();
	return 0;
}

void initCubePosition()
{
	for (int i = 0; i < ROWS; ++i)
	{
		for (int j = 0; j < COLS; ++j)
		{
			cubePositions[i][j] = glm::vec3(i, 0, j);
		}
	}
}

void initDataBuffers(unsigned int *VAO, unsigned int *VBO)
{
	float vertices[] = {
			-0.5f, 0.5f, -0.5f, 0.0f, 1.0f,
			0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
			0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
			0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
			-0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
			-0.5f, 0.5f, -0.5f, 0.0f, 1.0f
	};

	glGenVertexArrays(1, VAO);
	glGenBuffers(1, VBO);

	glBindVertexArray(*VAO);

	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) 0);
	glEnableVertexAttribArray(0);
	// texture coord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) (3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
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

void read_file()
{
	std::ifstream file("heads_ca.txt");
	if (file.is_open())
	{
		std::string line;
		int i = 0;
		std::string head_value_string;
		int number_end;
		while (getline(file, line))
		{
			int j = 0;
			while (line.size() != 0)
			{
				number_end = line.find(',');
				head_value_string = line.substr(0, number_end);
				graphData[i][j] = std::stof(head_value_string);
				line = line.substr(number_end + 2, line.size());
				j++;
			}
			i++;
		}
		file.close();
	}
}