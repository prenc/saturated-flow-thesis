#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <iostream>

#include <GL/gl.h>
#include "utils/shader.h"
#include "utils/glfwManager.h"
#include "params.h"

void processInput(GLFWwindow *window);

int SCR_WIDTH = 800;
int SCR_HEIGHT = 600;
Camera camera(glm::vec3(-2.0f, 0.0f, 0.0f));
void read_file();
float graphData[ROWS][COLS];

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{
	GLFWwindow *window = WindowCreator::createGLFWWindow(SCR_WIDTH, SCR_HEIGHT, &camera);
	read_file();
	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	Shader shader("att_vertex.glsl", "att_fragment.glsl");

	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	struct point
	{
		GLfloat x;
		GLfloat y;
		GLfloat z;
	};
	point vertices[ROWS][COLS];

	for (int i = 0; i < ROWS; ++i)
	{
		for (int j = 0; j < COLS; j++)
		{
			vertices[i][j].x = (float) (j - (COLS - 1) / 2) / (float) ((COLS - 1) / 2);
			vertices[i][j].y = (float) (i - (ROWS - 1) / 2) / (float) ((ROWS - 1) / 2);
			vertices[i][j].z = (graphData[i][j] - 50.0);
		}
	}

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	unsigned int EBO;
	glGenBuffers(1, &EBO);

	int indices[4 * ROWS * (COLS - 1)];
	int i = 0;

// Horizontal grid lines
	for (int y = 0; y < ROWS; y++)
	{
		for (int x = 0; x < COLS - 1; x++)
		{
			indices[i++] = y * COLS + x;
			indices[i++] = y * COLS + x + 1;
		}
	}

// Vertical grid lines
	for (int x = 0; x < ROWS; x++)
	{
		for (int y = 0; y < COLS - 1; y++)
		{
			indices[i++] = y * COLS + x;
			indices[i++] = (y + 1) * COLS + x;
		}
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glBindVertexArray(0);

	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shader.ID);
		glm::mat4 model = glm::mat4(1.0f);
		shader.setMat4("model", model);

		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f,
		                                        100.0f);
		shader.setMat4("projection", projection);


		glm::mat4 view = camera.GetViewMatrix();
		shader.setMat4("view", view);
		glm::mat4 texture_transform = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(1,1, 1)), glm::vec3(0,0, 0));

		glBindVertexArray(VAO);
		glDrawElements(GL_LINES, 4 * ROWS * (COLS - 1), GL_UNSIGNED_INT, 0);
/*		glPointSize(5.0);
		for (int i = 0; i < MESH_SIZE * MESH_SIZE; ++i)
		{
			glDrawArrays(GL_POINTS, i, 1);
		}*/

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);


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