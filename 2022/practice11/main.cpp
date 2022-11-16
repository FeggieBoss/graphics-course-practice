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
#include <random>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

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

layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_size;
layout (location = 2) in float in_rotation;

out float size;
out float rotation;
out vec3 position;

void main()
{
    gl_Position = vec4(in_position, 1.0);
    size = in_size;
    rotation = in_rotation;   
    position = in_position;
}
)";

const char geometry_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position; // task3
uniform float time; // task4

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in float size[]; // task1
in float rotation[]; // task5
in vec3 position[]; // task5

out vec2 texcoord; // task2

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;
    vec3 on_camera = camera_position - center;
    vec3 x = normalize(cross(on_camera, vec3(0.0, 1.0, 0.0)));
    vec3 y = normalize(cross(x, on_camera));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task5
    vec3 dop = x;
    x = x * cos(rotation[0]) + y * sin(rotation[0]);
    y = -dop * sin(rotation[0]) + y * cos(rotation[0]);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int i=1;i>=-1;i-=2) {
        for(int j=1;j>=-1;j-=2) {
            //gl_Position = projection * view * model * vec4(center + vec3(i*size[0],j*size[0],0.0), 1.0);

            //gl_Position = projection * view * model * vec4(center + x*i*size[0] + y*j*size[0], 1.0); // task3
            gl_Position = projection * view * model * vec4(center + x*i*size[0] + y*j*size[0], 1.0);

            texcoord = vec2(i,j);
            texcoord = vec2(0.5) + texcoord*0.5;

            EmitVertex();
        }
    }
    EndPrimitive();
}

)";

const char fragment_shader_source[] =
R"(#version 330 core

layout (location = 0) out vec4 out_color;

in vec2 texcoord; // task2
uniform sampler2D particle; // task7
uniform sampler1D color_texture; // task8

void main()
{
    out_color = vec4(1.0, 0.0, 0.0, 1.0);
    out_color = vec4(texcoord, 0.0, 1.0); // task2
    out_color = vec4(texcoord, 0.0, texture(particle, texcoord).r); // task7
    out_color = vec4(texture(color_texture, texture(particle, texcoord).r).rgb, texture(particle, texcoord).r); // task8
}
)";

GLuint create_shader(GLenum type, const char* source)
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

template <typename ... Shaders>
GLuint create_program(Shaders ... shaders)
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

struct particle
{
    glm::vec3 position;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task1
    float size = float(std::chrono::system_clock::now().time_since_epoch().count()%21)/100 + 0.17;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task5
    float rotation = 0.f;
    float angular_velocity = float(std::chrono::system_clock::now().time_since_epoch().count()%81)/100 + 0.1;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task4 
    glm::vec3 velocity = {0.f, 0.f, 0.f};
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

GLuint load_texture(std::string const& path)
{
    int width, height, channels;
    auto pixels = stbi_load(path.data(), &width, &height, &channels, 4);

    GLuint result;
    glGenTextures(1, &result);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels);

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
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 11",
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

    glClearColor(0.f, 0.f, 0.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto geometry_shader = create_shader(GL_GEOMETRY_SHADER, geometry_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, geometry_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint time_location = glGetUniformLocation(program, "time");
    GLuint velocity_location = glGetUniformLocation(program, "velocity");
    GLuint color_texture_location = glGetUniformLocation(program, "color_texture");
    GLuint particle_location = glGetUniformLocation(program, "particle");

    std::default_random_engine rng;

    // std::vector<particle> particles(256);
    // for (auto & p : particles)
    // {
    //     p.position.x = std::uniform_real_distribution<float>{-1.f, 1.f}(rng);
    //     p.position.y = 0.f;
    //     p.position.z = std::uniform_real_distribution<float>{-1.f, 1.f}(rng);
    // }
    std::vector<particle> particles;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(0));
    glEnableVertexAttribArray(1); // task3
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(12)); // task3
    glEnableVertexAttribArray(2); // task5
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(16)); // task5

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task7
    const std::string project_root = PROJECT_ROOT;
    const std::string particle_texture_path = project_root + "/particle.png";
    GLuint texture = load_texture(particle_texture_path);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // task8
    GLuint colors_texture;
    glGenTextures(1, &colors_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, colors_texture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    std::vector<glm::vec4> colors = {
      {0.f, 0.f, 1.f, 1.f}, // blue
      {1.f, 0.f, 0.f, 1.f}, // red
      {1.f, 1.f, 0.f, 1.f}, // yellow
      {1.f, 1.f, 1.f, 1.f}, // white
    };
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, colors.size(), 0, GL_RGBA, GL_FLOAT, colors.data());
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    glPointSize(5.f);

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_angle = 0.f;
    float camera_distance = 2.f;
    float camera_height = 0.5f;

    float camera_rotation = 0.f;

    glm::vec4 velocity = {0.f, 0.f, 0.f, 0.f};

    bool paused = false;

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
            if (event.key.keysym.sym == SDLK_SPACE)
                paused = !paused;
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
        if(!paused) {
            time += dt;
            
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // task6
            const float A = 0.5;
            const float C = 0.1;
            const float D = 0.6;

            while(!particles.empty() && particles.back().size < 0.05) particles.pop_back();
            if(particles.size()<1024) {
                particles.push_back(particle());

                particles.back().position.x = std::uniform_real_distribution<float>{-0.05, 0.05}(rng);
                particles.back().position.y = 0.f;
                particles.back().position.z = std::uniform_real_distribution<float>{-0.05, 0.05}(rng);
            }
            sort(particles.begin(),particles.end(), [](auto &a, auto &b) {return a.size > b.size;});

            for(auto &p : particles) {
                p.size *= exp(-D * dt);
                p.velocity += glm::vec3(0.f, A * dt, 0.f) * std::exp(-C * dt);
                p.position += p.velocity * dt;
                p.rotation += p.angular_velocity * dt;
            }
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }

        if (button_down[SDLK_UP])
            camera_distance -= 3.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 3.f * dt;

        if (button_down[SDLK_LEFT])
            camera_rotation -= 3.f * dt;
        if (button_down[SDLK_RIGHT])
            camera_rotation += 3.f * dt;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glEnable(GL_DEPTH_TEST);
        glDisable(GL_DEPTH_TEST); // task8
 
        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, -camera_height, -camera_distance});
        view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});

        glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        velocity.y += dt; // task4

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(particle), particles.data(), GL_STATIC_DRAW);

        glUseProgram(program);

        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        glUniform4fv(velocity_location, 1, reinterpret_cast<float *>(&velocity));
        glUniform1f(time_location, time);
        glUniform1i(color_texture_location, 1);
        glUniform1i(particle_location, 0);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, 4*particles.size());

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
