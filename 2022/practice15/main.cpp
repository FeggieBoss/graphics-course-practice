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
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <map>
#include <cmath>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include "msdf_loader.hpp"
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

const char msdf_vertex_shader_source[] =
    R"(#version 330 core

uniform mat4 transform;

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_texcoord;

out vec2 texcoord;

void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_Position = vec4(in_position, 0.0, 1.0); // task1
    gl_Position = transform * vec4(in_position, 0.0, 1.0); // task2

    texcoord = in_texcoord;
}
)";

const char msdf_fragment_shader_source[] =
    R"(#version 330 core

layout (location = 0) out vec4 out_color;

in vec2 texcoord;

uniform float sdfScale;
uniform sampler2D sdfTexture;

float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    out_color = vec4(0.0);
    out_color = vec4(texcoord, 0.f, 1.f); // task1
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task4
    float sdfTextureValue = median(texture(sdfTexture, texcoord).rgb);    
    float sdfValue = sdfScale * (sdfTextureValue - 0.5);
    float alpha = smoothstep(-0.5, 0.5, sdfValue);
    out_color = vec4(vec3(0.f), alpha);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task6
    float k = length(vec2(dFdx(sdfValue), dFdy(sdfValue)))/sqrt(2.0);
    alpha = smoothstep(-k, k, sdfValue);
    out_color = vec4(vec3(0.f), alpha);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task7
    out_color = vec4(1-alpha, 1-alpha, 1-alpha, 0.f);
    sdfValue = sdfScale * (sdfTextureValue - 0.5) + 1.f;
    k = length(vec2(dFdx(sdfValue), dFdy(sdfValue)))/sqrt(2.0);
    alpha = smoothstep(-k, k, sdfValue);
    out_color = out_color + vec4(0.f, 0.f, 0.f, alpha);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// task1
struct vertex
{
    glm::vec2 position;
    glm::vec2 texcoord;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint
create_shader(GLenum type, const char *source)
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

template <typename... Shaders>
GLuint create_program(Shaders... shaders)
{
    GLuint result = glCreateProgram();
    (glAttachShader(result, shaders), ...);
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

int main()
try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 15",
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

    auto msdf_vertex_shader = create_shader(GL_VERTEX_SHADER, msdf_vertex_shader_source);
    auto msdf_fragment_shader = create_shader(GL_FRAGMENT_SHADER, msdf_fragment_shader_source);
    auto msdf_program = create_program(msdf_vertex_shader, msdf_fragment_shader);

    GLuint transform_location = glGetUniformLocation(msdf_program, "transform");
    GLuint sdfScale_location = glGetUniformLocation(msdf_program, "sdfScale");
    GLuint sdfTexture_location = glGetUniformLocation(msdf_program, "sdfTexture");

    const std::string project_root = PROJECT_ROOT;
    const std::string font_path = project_root + "/font/font-msdf.json";

    auto const font = load_msdf_font(font_path);

    GLuint texture;
    int texture_width, texture_height;
    {
        int channels;
        auto data = stbi_load(font.texture_path.c_str(), &texture_width, &texture_height, &channels, 4);
        assert(data);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task1
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, texcoord));

    std::vector<vertex> vertices;

    // vertices = {
    //     {{0.f,0.f},{0.f,0.f}},
    //     {{100.f,0.f},{1.f,0.f}},
    //     {{0.f,100.f},{0.f,1.f}}
    // };
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), vertices.data(), GL_STATIC_DRAW);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    SDL_StartTextInput();

    std::map<SDL_Keycode, bool> button_down;

    std::string text = "Hello, world!";
    bool text_changed = true;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task5
    float min_x = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                switch (event.window.event)
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
                if (event.key.keysym.sym == SDLK_BACKSPACE && !text.empty())
                {
                    text.pop_back();
                    text_changed = true;
                }
                break;
            case SDL_TEXTINPUT:
                text.append(event.text.text);
                text_changed = true;
            case SDL_KEYUP:
                button_down[event.key.keysym.sym] = false;
                break;
            }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task3
        if (text_changed)
        {
            text_changed = false;

            min_x = std::numeric_limits<float>::infinity();
            max_x = -std::numeric_limits<float>::infinity();
            min_y = std::numeric_limits<float>::infinity();
            max_y = -std::numeric_limits<float>::infinity();

            vertices.clear();
            glm::vec2 pen = glm::vec2(0.f, 0.f);

            for (char c : text)
            {
                const msdf_font::glyph g = font.glyphs.at(c);

                std::vector<int> i_ = {0, 0, 1, 1, 0, 1};
                ;
                std::vector<int> j_ = {0, 1, 0, 0, 1, 1};
                for (int i = 0; i < 6; ++i)
                {
                    vertices.push_back({});
                    vertices.back().position = glm::vec2(g.xoffset + g.width * i_[i], g.yoffset + g.height * j_[i]) + pen;
                    vertices.back().texcoord = glm::vec2((g.x + g.width * i_[i]) * 1.f / texture_width, (g.y + g.height * j_[i]) * 1.f / texture_height);

                    ///////////////////////////////////////////////////////////////////////////////////////////////////////
                    // task5
                    min_x = std::min(min_x, vertices.back().position.x);
                    max_x = std::max(max_x, vertices.back().position.x);
                    min_y = std::min(min_y, vertices.back().position.y);
                    max_y = std::max(max_y, vertices.back().position.y);
                    ///////////////////////////////////////////////////////////////////////////////////////////////////////
                }
                pen.x += g.advance;
            }

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), vertices.data(), GL_STATIC_DRAW);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task1
        glUseProgram(msdf_program);
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task4
        glUniform1f(sdfScale_location, font.sdf_scale);
        glUniform1i(sdfTexture_location, 0);
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task2
        glm::mat4 transform = glm::transpose(glm::mat4{
            {2.f / (float)width, 0.f, 0.f, -1.f},
            {0.f, -2.f / (float)height, 0.f, 1.f},
            {0.f, 0.f, 1.f, 0.f},
            {0.f, 0.f, 0.f, 1.f},
        });
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&transform));
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task5
        // transform = glm::translate(transform, glm::vec3((width - min_x - max_x) / 2.f, (height - min_y - max_y) / 2.f, 0.f));
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&transform));
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task6
        transform = glm::translate(transform, glm::vec3((width - (min_x + max_x) * 5) / 2.f, (height - (min_y + max_y) * 5) / 2.f, 0.f));
        transform = glm::scale(transform, {5.f, 5.f, 1.f});
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&transform));
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // task1
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
