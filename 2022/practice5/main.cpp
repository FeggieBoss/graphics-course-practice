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
#include <cmath>

#include "obj_parser.hpp"
#include "stb_image.h"

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

uniform mat4 transform;
uniform mat4 projection;
uniform float dt;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 normal;
out vec2 texcoord;

void main()
{
    gl_Position = projection * transform * vec4(in_position, 1.0);
    normal = mat3(transform) * in_normal;
    texcoord = vec2(in_texcoord.x + dt, in_texcoord.y + dt);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D texture;

in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

void main()
{
    float lightness = 0.5 + 0.5 * dot(normalize(normal), normalize(vec3(1.0, 2.0, 3.0)));
    vec4 albedo_ = texture2D(texture, texcoord);
    vec3 albedo = vec3(albedo_.x, albedo_.y, albedo_.z);
    out_color = vec4(lightness * albedo, 1.0);
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

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 5",
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

    GLuint transform_location = glGetUniformLocation(program, "transform");
    GLuint projection_location = glGetUniformLocation(program, "projection");

    std::string project_root = PROJECT_ROOT;
    std::string cow_texture_path = project_root + "/cow.png";
    obj_data cow = parse_obj(project_root + "/cow.obj");

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task1
    GLuint vao;
    glGenVertexArrays(1, &vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);

    GLuint ebo;
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));
    glBufferData(GL_ARRAY_BUFFER, cow.vertices.size() * sizeof(obj_data::vertex), cow.vertices.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(3*4));
    glBufferData(GL_ARRAY_BUFFER, cow.vertices.size() * sizeof(obj_data::vertex), cow.vertices.data(), GL_STREAM_DRAW);
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // task2
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(3*4+3*4));
            glBufferData(GL_ARRAY_BUFFER, cow.vertices.size() * sizeof(obj_data::vertex), cow.vertices.data(), GL_STREAM_DRAW);
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cow.indices.size() * sizeof(std::uint32_t), cow.indices.data(), GL_STATIC_DRAW);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task3
    GLuint textureID=0;
    glGenTextures(1, &textureID);

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task 4
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    const int texture_size = 512;
    std::vector<std::uint32_t> pixels(texture_size * texture_size);
    for(int i=0;i<texture_size;++i) {
        for(int j=0;j<texture_size;++j) {
            pixels[i*texture_size+j] = (((i+j)&1) == 0 ? 0xFFFFFFFFu : 0xFF000000u);
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_size, texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    GLuint texture_location = glGetUniformLocation(program, "texture");
    glUseProgram(program);
    glUniform1i(texture_location, 0);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task4
    glGenerateMipmap(GL_TEXTURE_2D);

    std::vector<std::uint32_t> pixels_1(texture_size/2 * texture_size/2, 0xFFFF000u);
    glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8, texture_size/2, texture_size/2, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_1.data());


    std::vector<std::uint32_t> pixels_2(texture_size/4 * texture_size/4, 0xFF00FF00u);
    glTexImage2D(GL_TEXTURE_2D, 2, GL_RGBA8, texture_size/4, texture_size/4, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_2.data());


    std::vector<std::uint32_t> pixels_3(texture_size/8 * texture_size/8, 0xFF0000FFu);
    glTexImage2D(GL_TEXTURE_2D, 3, GL_RGBA8, texture_size/8, texture_size/8, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_3.data());
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task5
    GLuint textureID2=0;
    glGenTextures(1, &textureID2);

    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, textureID2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int texture2_width, texture2_height, texture2_nrChannels;
    unsigned char* pixels_texture2 = stbi_load(cow_texture_path.c_str(), &texture2_width, &texture2_height, &texture2_nrChannels, 4);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture2_width, texture2_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_texture2);

    glUseProgram(program);
    glUniform1i(texture_location, 1);

    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels_texture2);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task 6
    GLuint dt_location = glGetUniformLocation(program, "dt");
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    float angle_y = M_PI;
    float offset_z = -2.f;

    std::map<SDL_Keycode, bool> button_down;

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

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task 6
        glUseProgram(program);
        glUniform1f(dt_location, time);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (button_down[SDLK_UP]) offset_z -= 4.f * dt;
        if (button_down[SDLK_DOWN]) offset_z += 4.f * dt;
        if (button_down[SDLK_LEFT]) angle_y += 4.f * dt;
        if (button_down[SDLK_RIGHT]) angle_y -= 4.f * dt;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        float near = 0.1f;
        float far = 100.f;
        float top = near;
        float right = (top * width) / height;

        float transform[16] =
        {
            std::cos(angle_y), 0.f, -std::sin(angle_y), 0.f,
            0.f, 1.f, 0.f, 0.f,
            std::sin(angle_y), 0.f, std::cos(angle_y), offset_z,
            0.f, 0.f, 0.f, 1.f,
        };

        float projection[16] =
        {
            near / right, 0.f, 0.f, 0.f,
            0.f, near / top, 0.f, 0.f,
            0.f, 0.f, - (far + near) / (far - near), - 2.f * far * near / (far - near),
            0.f, 0.f, -1.f, 0.f,
        };

        glUseProgram(program);
        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform);
        glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task1
        glDrawElements(GL_TRIANGLES, cow.indices.size(), GL_UNSIGNED_INT, 0);
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
