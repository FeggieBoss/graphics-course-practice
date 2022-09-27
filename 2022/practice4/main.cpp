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
#include <chrono>
#include <vector>
#include <map>

#include "obj_parser.hpp"

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

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

out vec3 normal;

void main()
{
    gl_Position = view * transform * vec4(in_position, 1.0);
    normal = mat3(transform) * in_normal;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec3 normal;

layout (location = 0) out vec4 out_color;

void main()
{
    float lightness = 0.5 + 0.5 * dot(normalize(normal), normalize(vec3(1.0, 2.0, 3.0)));
    out_color = vec4(vec3(lightness), 1.0);
}
)";

GLuint create_shader(GLenum type, const char * source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 4",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint transform_location = glGetUniformLocation(program, "transform");

    std::string project_root = PROJECT_ROOT;
    obj_data bunny = parse_obj(project_root + "/bunny.obj");

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task1 
    
    //VBO
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bunny.vertices.size() * sizeof(obj_data::vertex), bunny.vertices.data(), GL_STREAM_DRAW);
    
    //VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, sizeof(obj_data::vertex), (void*)(3*4));

    // glEnableVertexAttribArray(2);
    // glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(6*4));

    //EBO
    GLuint ebo;
    glGenBuffers (1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, bunny.indices.size() * sizeof(std::uint32_t), bunny.indices.data(), GL_STATIC_DRAW);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task4 
    glEnable(GL_DEPTH_TEST);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task5
    float bunny_x = 0;
    float bunny_y = 0;

    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_WINDOWEVENT: switch (event.window.event)
            {
            case SDL_WINDOWEVENT_RESIZED:
                width = event.window.data1;
                height = event.window.data2;
                glViewport(0, 0, width, height);
                break;
            }
            break;
        case SDL_KEYDOWN:
            button_down[event.key.keysym.sym] = true;
            break;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        glClear(GL_COLOR_BUFFER_BIT);

        // float view[16] =
        // {
        //     1.f, 0.f, 0.f, 0.f,
        //     0.f, 1.f, 0.f, 0.f,
        //     0.f, 0.f, 1.f, 0.f,
        //     0.f, 0.f, 0.f, 1.f,
        // };

        // float transform[16] =
        // {
        //     1.f, 0.f, 0.f, 0.f,
        //     0.f, 1.f, 0.f, 0.f,
        //     0.f, 0.f, 1.f, 0.f,
        //     0.f, 0.f, 0.f, 1.f,
        // };

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task3
        float near = 0.1;
        float far = 100;
        float right = near * tan(90/2);
        float left = -right;
        float top = height*right/width;
        float bottom = -top;

        float view[16] =
        {
            2.f * near / (right -left), 0.f, (right+left)/(right-left), 0.f,
            0.f, 2.f * near / (top-bottom), (top+bottom)/(top-bottom), 0.f,
            0.f, 0.f, -(far+near)/(far-near), -(2*far*near)/(far-near),
            0.f, 0.f, -1.f, 0,
        };
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task2
        float angle = time;
        float scale = 0.5f;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task5
        float speed = 1.5;
        if(button_down[SDLK_LEFT])
            bunny_x -= speed * dt;
        if(button_down[SDLK_RIGHT])
            bunny_x += speed * dt;
        if(button_down[SDLK_UP])
            bunny_y += speed * dt;
        if(button_down[SDLK_DOWN])
            bunny_y -= speed * dt;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // task2
        float transform1[16] =
        {
            cos(time) * scale, 0.f, -sin(time) * scale, bunny_x + 0.f,
            0.f, scale, 0.f, bunny_y + 0.f,
            sin(time) * scale, 0.f, cos(time) * scale, -3.f,
            0.f, 0.f, 0.f, 1.f,
        };
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////


        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform1);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task6
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task4
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task1
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, 0);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task7
        float base0_x=-0.9, base0_y=-0.9, base0_z = -2.f, scale0 = 0.5;
        float transform0[16] =
        {
            cos(time) * scale0, -sin(time) * scale0,  0.f, base0_x + bunny_x,
            sin(time) * scale0, cos(time) * scale0, 0.f, base0_y + bunny_y,
            0.f, 0.f, scale0, base0_z + -3.f,
            0.f, 0.f, 0.f, 1.f,
        };

        float base2_x=0.7, base2_y=0.2, base2_z = 2.5f, scale2 = 0.1;
        float transform2[16] =
        {
            scale2, 0.f, 0.f, base2_x + bunny_x,
            0.f, cos(time)*scale2, -sin(time)*scale2, base2_y + bunny_y,
            0.f, sin(time)*scale2, cos(time)*scale2, base2_z + -3.f,
            0.f,0.f,0.f,1.f
        };

        glCullFace(GL_BACK);

        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform0);
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, 0);

        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform2);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, 0);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
