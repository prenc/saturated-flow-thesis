#version 330 core
in vec4 graph_coord;
void main()
{
    gl_FragColor = vec4(0.0, 0.0, 1 - (graph_coord.y + 0.8) , 1.0f);
}