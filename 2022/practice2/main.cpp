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

/* TASK1-5
const vec2 VERTICES[3] = vec2[3](
    vec2(0.0, 1.0),
    vec2(-sqrt(0.75), -0.5),
    vec2( sqrt(0.75), -0.5)
);
const vec3 COLORS[3] = vec3[3](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);
*/

// TASK6 
const vec2 center = vec2(0.0, 0.0);
const float rotation_angle = 60;
const float rotation_angle_rad = rotation_angle*acos(-1)/180;

const vec3 COLORS[8] = vec3[8](
    vec3(0.5, 0.5, 0.5),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0)
);

out vec3 color;

uniform float scale;
uniform float angle;
uniform mat4 transform;
uniform mat4 view;

void main()
{
    /* TASK1-5
    vec2 position = VERTICES[gl_VertexID];
    */
    vec2 position = center;
    if(gl_VertexID > 0) {
        position = vec2(cos((gl_VertexID-1)*rotation_angle_rad), sin((gl_VertexID-1)*rotation_angle_rad));
    }

    /*
    TASKS 1 and 2
    
    // task1 : 
    gl_Position = vec4(position.x * scale, position.y * scale, 0.0, 1.0);


    // task2 : 
    mat4x4 rotation = mat4x4(
        cos(angle), -sin(angle), 0, 0,
        sin(angle), cos(angle), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
    gl_Position = vec4(rotation * gl_Position);
    */
    
    gl_Position = view * transform * vec4(position.x, position.y, 0.0, 1.0);
    
    color = COLORS[gl_VertexID];
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec3 color;

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(color, 1.0);
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 2",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

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

    /////////////////////////////////////////////////////////////////////////////////////
    // task6 - off VSync
    SDL_GL_SetSwapInterval(0);
    /////////////////////////////////////////////////////////////////////////////////////

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    GLuint vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);

    GLuint program = create_program(vertex_shader, fragment_shader);

    /////////////////////////////////////////////////////////////////////////////////////
    // task1 - setting scale
    const char* name_of_uniform_scale = "scale";
    glUseProgram(program);
    glUniform1f(glGetUniformLocation(program, name_of_uniform_scale), 0.5);
    /////////////////////////////////////////////////////////////////////////////////////
    // task2 - getting angle id
    const char* name_of_uniform_angle = "angle";
    float time = 0.f;
    GLint id_uniform_angle = glGetUniformLocation(program, name_of_uniform_angle);
    /////////////////////////////////////////////////////////////////////////////////////
    // task3 - getting transform id
    const char* name_of_uniform_transform = "transform";
    GLint id_uniform_transform = glGetUniformLocation(program, name_of_uniform_transform);
    /////////////////////////////////////////////////////////////////////////////////////
    // task5 - getting view id
    const char* name_of_uniform_view = "view";
    GLint id_uniform_view = glGetUniformLocation(program, name_of_uniform_view);
    /////////////////////////////////////////////////////////////////////////////////////

    GLuint vao;
    glGenVertexArrays(1, &vao);

    auto last_frame_start = std::chrono::high_resolution_clock::now();

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
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        /////////////////////////////////////////////////////////////////////////////////////
        // task6 - changing dt
        //float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        float dt = 0.016f;
        /////////////////////////////////////////////////////////////////////////////////////
        last_frame_start = now;

        glClear(GL_COLOR_BUFFER_BIT);

        /////////////////////////////////////////////////////////////////////////////////////
        // task 2 - setting angle
        time += dt;

        // std::cout<<dt<<std::endl; task6
        
        glUseProgram(program);
        glUniform1f(id_uniform_angle, time);
        /////////////////////////////////////////////////////////////////////////////////////
        // task3 - setting transform
        float transform[16] = {
            0.5f * cos(time), -0.5f * sin(time), 0, 0,
            0.5f * sin(time), 0.5f * cos(time), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        glUniformMatrix4fv(id_uniform_transform, 1, GL_TRUE, transform);
        /////////////////////////////////////////////////////////////////////////////////////
        // task4 - adding shift to transform
        float x = 0.6 * cos(time);
        float y = 0.6 * sin(time);

        /* need to calc mult(A,transform)
        A = {
            1,0,0,x
            0,1,0,y
            0,0,1,0
            0,0,0,1
        }

        */
        float transform_and_shift[16] = {
            0.5f * cos(time), -0.5f * sin(time), 0, x,
            0.5f * sin(time), 0.5f * cos(time), 0, y,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        glUniformMatrix4fv(id_uniform_transform, 1, GL_TRUE, transform_and_shift);
        /////////////////////////////////////////////////////////////////////////////////////
        // task5 - setting view
        float aspect_ratio = 1.0f * width / height;     
        float view[16] = {
            1.0f/aspect_ratio, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        glUniformMatrix4fv(id_uniform_view, 1, GL_TRUE, view);
        /////////////////////////////////////////////////////////////////////////////////////

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 8);

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
