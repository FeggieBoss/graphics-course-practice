#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>


const std::string fita = "fitusya";
const std::string code1 = 
R"(#version 330 core

const vec2 VERTICES[3] = vec2[3](
	vec2(0.0, 0.0),
	vec2(1.0, 0.0),
	vec2(0.0, 1.0)
);

out vec3 color;
out float x;
out float y;

void main()
{
	gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);
	x = gl_Position.x;
	y = gl_Position.y;
	color = vec3(x,y,0.0);
}
)";
const std::string code2 = 
R"(#version 330 core
layout (location = 0) out vec4 out_color;

in vec3 color;
in float x;
in float y;

void main()
{
	// out_color = vec4(color, 1.0); gradient task
	float chess_color = (int(floor(x*16))&1) ^ ((int(floor(y*16))&1));
	out_color = vec4(chess_color, chess_color, chess_color, 1.0);
}
)";

std::string to_string(std::string_view str)
{
	return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
	throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
	throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}


GLuint create_shader(GLenum shader_type, const char* shader_source) {
	GLuint shader = glCreateShader(shader_type);
	
	if(shader==0) {
		std::cerr<<fita+(" says: create shader error")<<std::endl;
		return EXIT_FAILURE;
	}

	glShaderSource(shader, 1, &shader_source, nullptr);
	glCompileShader(shader);

	GLint shader_compile_status;
	glGetShaderiv(shader,GL_COMPILE_STATUS, &shader_compile_status);

	if(shader_compile_status==GL_FALSE) {
		GLsizei length_info_log;
		GLsizei buffer_size;

		std::string info_log_shader(buffer_size, '\0');
		glGetShaderiv(shader,GL_INFO_LOG_LENGTH, &buffer_size);

		glGetShaderInfoLog(shader,buffer_size,&length_info_log, const_cast<GLchar*>(info_log_shader.c_str()));

		throw std::runtime_error(fita+" says: "+info_log_shader);
	}

	return shader;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
	GLuint program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);
	
	GLint program_link_status;
	glGetShaderiv(program,GL_LINK_STATUS, &program_link_status);

	if(program_link_status==GL_FALSE) {
		GLsizei length_info_log;
		GLsizei buffer_size;

		std::string info_log_program(buffer_size, '\0');
		glGetShaderiv(program,GL_INFO_LOG_LENGTH, &buffer_size);

		glGetShaderInfoLog(program,buffer_size,&length_info_log, const_cast<GLchar*>(info_log_program.c_str()));

		throw std::runtime_error(fita+" says: "+info_log_program);
	}

	return program;
}

int main() try
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		sdl2_fail("SDL_Init: ");

	SDL_Window * window = SDL_CreateWindow("Graphics course practice 1",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		800, 600,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

	if (!window)
		sdl2_fail("SDL_CreateWindow: ");

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	SDL_GLContext gl_context = SDL_GL_CreateContext(window);
	if (!gl_context)
		sdl2_fail("SDL_GL_CreateContext: ");

	if (auto result = glewInit(); result != GLEW_NO_ERROR)
		glew_fail("glewInit: ", result);

	if (!GLEW_VERSION_3_3)
		throw std::runtime_error("OpenGL 3.3 is not supported");

	glClearColor(0.8f, 0.8f, 1.f, 0.f);











	//
	auto program = create_program(
		create_shader(GL_VERTEX_SHADER, code1.c_str()),
		create_shader(GL_FRAGMENT_SHADER, code2.c_str())
	);

	GLuint array;
	glGenVertexArrays(1,&array);

	glUseProgram(program);
	glBindVertexArray(array);		

	glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
	//














	bool running = true;
	while (running)
	{
		for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
		{
		case SDL_QUIT:
			running = false;
			break;
		}

		if (!running)
			break;

		glClear(GL_COLOR_BUFFER_BIT);






		//
		glDrawArrays(GL_TRIANGLES, 0, 3);
		//







		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
