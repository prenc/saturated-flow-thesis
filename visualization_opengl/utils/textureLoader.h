//
// Created by pecatoma on 28.11.2019.
//

#ifndef VISUALIZATION_OPENGL_TEXTURELOADER_H
#define VISUALIZATION_OPENGL_TEXTURELOADER_H

#include <glad/glad.h>
#include <iostream>
#include <cstring>
#include "../libs/stb_image.h"

unsigned int checkInputColorType(const char *path, int pathSize);
int checkInputType(const char *path, int pathSize);

const int EXTENSION_SIZE = 3;
enum EXTENSION_TYPES
{
	JPG = 1, PNG = 2
};

void loadTexture(unsigned int *texture, const char *texturePath, int pathSize)
{
	glGenTextures(1, texture);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, *texture);

	// set the texture wrapping/filtering options (on the currently bound texture object)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	int width, height, nrChannels;
	stbi_set_flip_vertically_on_load(true);
	unsigned char *data = stbi_load(texturePath, &width, &height, &nrChannels, 0);
	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, checkInputColorType(texturePath, pathSize),
		             GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
}


unsigned int checkInputColorType(const char *path, int pathSize)
{
	int type = checkInputType(path, pathSize);
	switch (type)
	{
		case JPG:
			return GL_RGB;
		case PNG:
			return GL_RGBA;
		default:
			return -1;
	}
}

int checkInputType(const char *path, int pathSize)
{
	char extension[EXTENSION_SIZE];
	memcpy(extension, &path[pathSize - EXTENSION_SIZE - 1], EXTENSION_SIZE);
	if (strcmp(extension, "jpg") == 0)
	{ return JPG; }
	if (strcmp(extension, "png") == 0)
	{ return PNG; }
	return -1;
}

#endif //VISUALIZATION_OPENGL_TEXTURELOADER_H
