//
// Created by pecatoma on 02.12.2019.
//

#include "Chart.h"

Chart::Chart(std::string outputPath, Camera *camera, int scr_width, int scr_height)
{
	this->outputPath = outputPath;
	this->camera = camera;
	this->scr_width = scr_width;
	this->scr_height = scr_height;
	this->shader = Shader(vertexShaderPath.c_str(),fragmentShaderPath.c_str());
	init_chart_data();
}

void Chart::draw()
{
	glUseProgram(shader.ID);
	glm::mat4 model = glm::mat4(1.0f);
	shader.setMat4("model", model);

	glm::mat4 projection = glm::perspective(glm::radians(camera->Zoom), (float) scr_width / (float) scr_height, 0.1f,
	                                        100.0f);
	shader.setMat4("projection", projection);

	glm::mat4 view = camera->GetViewMatrix();
	shader.setMat4("view", view);
	glm::mat4 texture_transform = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(1,1, 1)), glm::vec3(0,0, 0));

	glBindVertexArray(VAO);
	glDrawElements(GL_LINES, 4 * ROWS * (COLS - 1), GL_UNSIGNED_INT, 0);
}

void Chart::init_chart_data()
{
	read_simulation_output_file();

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
	//init_buffers();
}

void Chart::init_buffers()
{
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	init_position_VBO();
	init_position_EBO();
	glBindVertexArray(0);
}

void Chart::init_position_VBO()
{
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	put_data_positions_in_VBO();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Chart::init_position_EBO()
{
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

	put_data_indexes_in_EBO();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Chart::put_data_positions_in_VBO()
{
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
}

void Chart::put_data_indexes_in_EBO()
{

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
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

void Chart::delete_buffers()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}
void Chart::read_simulation_output_file()
{
	std::ifstream file(outputPath);
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