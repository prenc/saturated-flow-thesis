#version 330 core
in vec4 graph_coord;
void main()
{
    gl_FragColor = graph_coord / 2.0 + 0.5;
}