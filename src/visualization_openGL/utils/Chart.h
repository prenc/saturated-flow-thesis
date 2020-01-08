//
// Created by pecatoma on 02.12.2019.
//

#ifndef VISUALIZATION_OPENGL_CHART_H
#define VISUALIZATION_OPENGL_CHART_H


#include <glad/glad.h>
#include "shader.h"
#include "camera.h"
#include <fstream>
#include <string>
#include "../params.h"

class Chart
{
public:
	Chart(std::string outputPath, Camera *camera, int scr_width, int scr_height);
	void draw();
	void delete_buffers();
private:
	unsigned int VBO, VAO, EBO;
	std::string outputPath;
	Camera *camera;
	float graphData[ROWS][COLS];
	struct point
	{
		GLfloat x;
		GLfloat y;
		GLfloat z;
	};
	std::string fragmentShaderPath = "shaders/chart_f.glsl";
	std::string vertexShaderPath = "shaders/chart_v.glsl";
	Shader shader = Shader();
	int scr_width, scr_height;

	void read_simulation_output_file();

	void init_chart_data();

	void init_buffers();

	void init_position_VBO();

	void put_data_indexes_in_EBO();

	void init_position_EBO();

	void put_data_positions_in_VBO();

};


#endif //VISUALIZATION_OPENGL_CHART_H
