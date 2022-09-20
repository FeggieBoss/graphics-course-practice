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
uniform float shift;

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec4 in_color;
layout (location = 2) in float dist;

out vec4 color;
out float inter_dist;

void main()
{
    gl_Position = view * vec4(in_position, 0.0, 1.0);
    color = in_color;
    inter_dist = dist + shift;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core
uniform int in_is_drawing;

in vec4 color;
in float inter_dist;

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = color;
    if(in_is_drawing==0 && mod(inter_dist, 40.0) < 20.0) {
        discard;
    }
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

struct vec2
{
    float x;
    float y;
};

struct vertex
{
    vec2 position;
    std::uint8_t color[4];

    float dist; // task6
};

vec2 bezier(std::vector<vertex> const & vertices, float t)
{
    std::vector<vec2> points(vertices.size());

    for (std::size_t i = 0; i < vertices.size(); ++i)
        points[i] = vertices[i].position;

    // De Casteljau's algorithm
    for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
        for (std::size_t i = 0; i + k + 1 < vertices.size(); ++i) {
            points[i].x = points[i].x * (1.f - t) + points[i + 1].x * t;
            points[i].y = points[i].y * (1.f - t) + points[i + 1].y * t;
        }
    }
    return points[0];
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 3",
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

    SDL_GL_SetSwapInterval(0);

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint view_location = glGetUniformLocation(program, "view");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task6 для двжигающегося пунктира
    GLuint shift_location = glGetUniformLocation(program, "shift");
    GLuint flag_location = glGetUniformLocation(program, "in_is_drawing");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;   

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task1 
    std::vector<vertex> vertices = std::vector<vertex>({
        {{0,0},{255,0,0},1},
        {{0,0.5},{0,255,0},1},
        {{0.5,0},{0,0,255},1}
    });
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task3
    vertices = std::vector<vertex>({
        {{0,0},{255,0,0,1},0},
        {{0,height},{0,255,0,1},0},
        {{width,0},{0,0,255,1},0}
    });
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), vertices.data(), GL_STREAM_DRAW);

    vertex data = vertices[2];
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertex), &data);
    std::cout<< sizeof(vertex) << " " << int(data.position.x)<<std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task2 
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(8));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(8+4));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task4
    std::vector<vertex> vertices2;
    GLuint vbo2;
    glGenBuffers(1, &vbo2);
    glBindBuffer(GL_ARRAY_BUFFER, vbo2);    

    GLuint vao2;
    glGenVertexArrays(1, &vao2);
    glBindVertexArray(vao2);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(8));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(8+4));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task6
    int quality = 4;

    std::vector<vertex> vertices3;
    GLuint vbo3;
    glGenBuffers(1, &vbo3);
    glBindBuffer(GL_ARRAY_BUFFER, vbo3);    

    GLuint vao3;
    glGenVertexArrays(1, &vao3);
    glBindVertexArray(vao3);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(8));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(8+4));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool running = true;
    while (running)
    {
        bool is_changed2 = false, is_changed3 = false;

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
        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                is_changed2 = true; //task6

                int mouse_x = event.button.x;
                int mouse_y = event.button.y;

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // task4
                vertices2.push_back({
                    {mouse_x,mouse_y},
                    {0,0,0,1},
                    {0}
                });
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            }
            else if (event.button.button == SDL_BUTTON_RIGHT)
            {
                is_changed2 = true; //task6
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // task4
                vertices2.pop_back();
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            }
            break;
        case SDL_KEYDOWN:
            if (event.key.keysym.sym == SDLK_LEFT) {
                is_changed3 = true;
                --quality;
                quality=std::max(quality,1);
            }
            else if (event.key.keysym.sym == SDLK_RIGHT) {
                is_changed3 = true;
                ++quality;
            }
            break;
        }
        is_changed3 = std::max(is_changed3, is_changed2);

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        glClear(GL_COLOR_BUFFER_BIT);

        float view[16] =
        {
            2.0f/width, 0.f, 0.f, -1.f,
            0.f, -2.0f/height, 0.f, 1.0f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };

        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task6 setting shift
        glUniform1f(shift_location, time*100);
        glUniform1i(flag_location, 1);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task2
        // glDrawArrays(GL_TRIANGLES, 0, 3);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task4,5
        glPointSize(10);
        glLineWidth(5.f);
        if (is_changed2) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo2);
            glBindVertexArray(vao2);
            glBufferData(GL_ARRAY_BUFFER, vertices2.size() * sizeof(vertex), vertices2.data(), GL_STREAM_DRAW);

            glDrawArrays(GL_LINE_STRIP, 0, vertices2.size());
            glDrawArrays(GL_POINTS, 0, vertices2.size());
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task6
        glBindBuffer(GL_ARRAY_BUFFER, vbo3);
        glBindVertexArray(vao3);
        if (is_changed3) {
            vertices3.clear();

            int n = quality*vertices2.size()-1;
            for(int i=0;i<=n;++i) {
                auto p = bezier(vertices2, 1.f*i/n);
                vertices3.push_back(
                    {
                        {p.x,p.y},
                        {255,0,255,1}, 
                        (vertices3.empty()?0:vertices3.back().dist + std::hypot(vertices3.back().position.x - p.x, vertices3.back().position.y - p.y))
                    }
                );
            }

            glBufferData(GL_ARRAY_BUFFER, vertices3.size() * sizeof(vertex), vertices3.data(), GL_STREAM_DRAW);
        }
        glUniform1i(flag_location, 0);
        glDrawArrays(GL_LINE_STRIP, 0, vertices3.size());
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
