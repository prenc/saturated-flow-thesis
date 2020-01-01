#version 330 core
layout (location = 0) in vec3 aPos;
out vec4 graph_coord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    graph_coord =  vec4(aPos.x, aPos.z, aPos.y, 1);

    gl_Position = projection * view * model * vec4(aPos.x,aPos.z,aPos.y, 1.0);
}