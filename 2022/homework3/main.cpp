#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
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

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_tangent;
layout (location = 2) in vec3 in_normal;
layout (location = 3) in vec2 in_texcoord;

out vec3 position;
out vec3 normal;
out vec2 texcoord;

void main()
{
    position = (model * vec4(in_position, 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
    normal = normalize(mat3(model) * in_normal);
    texcoord = in_texcoord;
}
)";

const char fragment_shader_source[] =
    R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;


uniform sampler2D snow_texture;
uniform sampler2D environment_texture;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

const float PI = acos(-1.f);

float diffuse(vec3 normal, vec3 light_direction) {
    return max(0.0, dot(normal, light_direction));
}

void main()
{
    float ambient_light = 0.2;

    vec3 camera_direction = normalize(position - camera_position);
    float cos_theta = abs(dot(normal, camera_direction));
    float n = 1.5f;
    float R0 = pow((1-n)/(1+n), 2.f);
    float R = R0 + (1-R0)*pow(1-cos_theta, 5.f);

    vec3 reflected_direction = reflect(camera_direction, normal);
    float lightness = ambient_light + diffuse(normal, light_direction);
    float x = atan(reflected_direction.z, reflected_direction.x) / PI * 0.5 + 0.5;
    float y = -atan(reflected_direction.y, length(reflected_direction.xz)) / PI + 0.5;
    vec3 reflec_color = texture(environment_texture,vec2(x,y)).xyz;

    out_color = vec4(lightness * reflec_color, R);
}
)";

const char background_vertex_shader_source[] =
    R"(#version 330 core
uniform mat4 view_projection_inverse;

const vec2 VERTICES[6] = vec2[6](
	vec2(-1.0, -1.0),
	vec2(-1.0, 1),
	vec2(1.0, 1.0),
    vec2(1.0, 1.0),
	vec2(1.0, -1.0),
	vec2(-1.0, -1.0)
);
out vec3 position;

void main()
{
    gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);

    vec4 clip_space = view_projection_inverse * gl_Position;
    position = clip_space.xyz / clip_space.w;
}
)";

const char background_fragment_shader_source[] =
    R"(#version 330 core
uniform sampler2D environment_texture;

uniform vec3 camera_position;

in vec3 position;

layout (location = 0) out vec4 out_color;


const float PI = 3.141592653589793;

void main() 
{
    float ambient_light = 0.9;

    vec3 view_direction = position-camera_position;
    
    float x = atan(view_direction.z, view_direction.x) / PI * 0.5 + 0.5;
    float y = -atan(view_direction.y, length(view_direction.xz)) / PI + 0.5;
    
    out_color = vec4(ambient_light * texture(environment_texture,vec2(x,y)).xyz, 0.f);
}
)";

const char floor_vertex_shader_source[] =
    R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_tangent;
layout (location = 2) in vec3 in_normal;
layout (location = 3) in vec2 in_texcoord;

out vec3 position;
out vec3 normal;
out vec2 texcoord;

void main()
{
    position = (model * vec4(in_position, 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
    normal = normalize(mat3(model) * in_normal);
    texcoord = in_texcoord;
}
)";

const char floor_fragment_shader_source[] =
    R"(#version 330 core

uniform vec3 light_direction;
uniform vec3 camera_position;


uniform sampler2D snow_texture;
uniform sampler2D normal_texture;
uniform sampler2D environment_texture;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(texture(snow_texture,texcoord).rgb, 1.0);
}
)";

const char watch_tower_vertex_shader_source[] =
    R"(#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
out vec3 position;
out vec3 normal;
out vec2 texcoord;
void main()
{
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    position = (model * vec4(in_position, 1.0)).xyz;
    normal = normalize((model * vec4(in_normal, 0.0)).xyz);
    texcoord = vec2(in_texcoord.x, 1.f-in_texcoord.y);
}
)";

const char watch_tower_fragment_shader_source[] =
    R"(#version 330 core
uniform vec3 ambient;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 camera_position;
uniform mat4 transform;
uniform sampler2D shadow_map;
uniform sampler2D texture;

in vec3 position;
in vec3 normal;
in vec2 texcoord;
layout (location = 0) out vec4 out_color;
float diffuse(vec3 direction) {
    return max(0.0, dot(normal, direction));
}
void main()
{
    vec3 light = vec3(1.f);
    vec3 color = texture2D(texture, texcoord).xyz * light;
    out_color = vec4(color, 1.0);
}
)";

GLuint create_shader(GLenum type, const char *source)
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

struct vertex
{
    glm::vec3 position;
    glm::vec3 tangent;
    glm::vec3 normal;
    glm::vec2 texcoords;
};

std::pair<std::vector<vertex>, std::vector<std::uint32_t>> generate_sphere(float radius, int quality)
{
    std::vector<vertex> vertices;

    for (int latitude = -quality; latitude <= quality; ++latitude)
    {
        for (int longitude = 0; longitude <= 4 * quality; ++longitude)
        {
            float lat = (latitude * glm::pi<float>()) / (2.f * quality);
            float lon = (longitude * glm::pi<float>()) / (2.f * quality);

            auto &vertex = vertices.emplace_back();
            vertex.normal = {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
            vertex.position = vertex.normal * radius;
            vertex.tangent = {-std::cos(lat) * std::sin(lon), 0.f, std::cos(lat) * std::cos(lon)};
            vertex.texcoords.x = (longitude * 1.f) / (4.f * quality);
            vertex.texcoords.y = (latitude * 1.f) / (2.f * quality) + 0.5f;
        }
    }

    std::vector<std::uint32_t> indices;

    for (int latitude = 0; latitude < 2 * quality; ++latitude)
    {
        for (int longitude = 0; longitude < 4 * quality; ++longitude)
        {
            std::uint32_t i0 = (latitude + 0) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i1 = (latitude + 1) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i2 = (latitude + 0) * (4 * quality + 1) + (longitude + 1);
            std::uint32_t i3 = (latitude + 1) * (4 * quality + 1) + (longitude + 1);

            indices.insert(indices.end(), {i0, i1, i2, i2, i1, i3});
        }
    }

    return {std::move(vertices), std::move(indices)};
}
std::pair<std::vector<vertex>, std::vector<std::uint32_t>> generate_half_sphere(float radius, int quality)
{
    const float k = 0.9f;

    std::vector<vertex> vertices;
    for (int latitude = 0; latitude <= quality; ++latitude)
    {
        for (int longitude = 0; longitude <= 4 * quality; ++longitude)
        {
            float lat = (latitude * glm::pi<float>()) / (2.f * quality) * k - glm::pi<float>() / 2.f;
            float lon = (longitude * glm::pi<float>()) / (2.f * quality);
            auto &vertex = vertices.emplace_back();

            vertex.normal = {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
            vertex.position = vertex.normal * radius;
            vertex.tangent = {-std::cos(lat) * std::sin(lon), 0.f, std::cos(lat) * std::cos(lon)};
            vertex.texcoords.x = (longitude * 1.f) / (4.f * quality);
            vertex.texcoords.y = (latitude * 1.f) / (2.f * quality) + 0.5f;
        }
    }

    int quality2 = quality * 10;
    for (int latitude = quality2; latitude <= quality2; ++latitude)
    {
        for (int longitude = 0; longitude <= 4 * quality2 + 1; ++longitude)
        {
            float lat = (latitude * glm::pi<float>()) / (2.f * quality2) * k - glm::pi<float>() / 2.f;
            float lon = (longitude * glm::pi<float>()) / (2.f * quality2);

            auto &vertex = vertices.emplace_back();

            vertex.normal = {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
            vertex.position = vertex.normal * radius;
            vertex.tangent = {-std::cos(lat) * std::sin(lon), 0.f, std::cos(lat) * std::cos(lon)};

            vertex.normal = glm::vec3(0.f, 1.f, 0.f);

            if (longitude > 4 * quality2)
            {
                vertex.position = glm::vec3(0.f, std::sin(lat), 0.f) * radius;
            }

            vertex.texcoords = (glm::vec2({vertex.position.z, vertex.position.x})) * 0.5f;

            if (longitude > 4 * quality2)
                vertex.texcoords = glm::vec2(0.5f, 0.5f);
        }
    }

    std::vector<std::uint32_t> indices;
    for (int latitude = 0; latitude < quality; ++latitude)
    {
        for (int longitude = 0; longitude < 4 * quality; ++longitude)
        {
            std::uint32_t i0 = (latitude + 0) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i1 = (latitude + 1) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i2 = (latitude + 0) * (4 * quality + 1) + (longitude + 1);
            std::uint32_t i3 = (latitude + 1) * (4 * quality + 1) + (longitude + 1);

            indices.insert(indices.end(), {i0, i1, i2, i2, i1, i3});
        }
    }
    for (int longitude = 0; longitude < 4 * quality2; ++longitude)
    {
        std::uint32_t i0 = vertices.size() - 1;
        std::uint32_t i1 = (quality + 1) * (4 * quality + 1) + (longitude + 0);
        std::uint32_t i2 = (quality + 1) * (4 * quality + 1) + (longitude + 1);
        indices.insert(indices.end(), {i0, i1, i2});
    }
    return {std::move(vertices), std::move(indices)};
}

GLuint load_texture(std::string const &path)
{
    int width, height, channels;
    auto pixels = stbi_load(path.data(), &width, &height, &channels, 4);

    GLuint result;
    glGenTextures(1, &result);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels);

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
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 5",
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

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint snow_texture_location = glGetUniformLocation(program, "snow_texture");
    GLuint environment_texture_location = glGetUniformLocation(program, "environment_texture");

    GLuint background_vao;
    glGenVertexArrays(1, &background_vao);

    auto background_vertex_shader = create_shader(GL_VERTEX_SHADER, background_vertex_shader_source);
    auto background_fragment_shader = create_shader(GL_FRAGMENT_SHADER, background_fragment_shader_source);
    auto background_program = create_program(background_vertex_shader, background_fragment_shader);

    GLuint view_projection_inverse_location = glGetUniformLocation(background_program, "view_projection_inverse");
    GLuint background_camera_position_location = glGetUniformLocation(background_program, "camera_position");
    GLuint background_environment_texture_location = glGetUniformLocation(background_program, "environment_texture");

    GLuint floor_vao, floor_vbo, floor_ebo;
    glGenVertexArrays(1, &floor_vao);
    glBindVertexArray(floor_vao);
    glGenBuffers(1, &floor_vbo);
    glGenBuffers(1, &floor_ebo);

    auto floor_vertex_shader = create_shader(GL_VERTEX_SHADER, floor_vertex_shader_source);
    auto floor_fragment_shader = create_shader(GL_FRAGMENT_SHADER, floor_fragment_shader_source);
    auto floor_program = create_program(floor_vertex_shader, floor_fragment_shader);

    GLuint floor_snow_texture_location = glGetUniformLocation(floor_program, "snow_texture");
    GLuint floor_model_location = glGetUniformLocation(program, "model");
    GLuint floor_view_location = glGetUniformLocation(program, "view");
    GLuint floor_projection_location = glGetUniformLocation(program, "projection");
    GLuint half_sphere_index_count;
    {
        auto [vertices, indices] = generate_half_sphere(0.95f, 16);

        glBindBuffer(GL_ARRAY_BUFFER, floor_vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, floor_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        half_sphere_index_count = indices.size();
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, tangent));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, texcoords));

    GLuint sphere_vao, sphere_vbo, sphere_ebo;
    glGenVertexArrays(1, &sphere_vao);
    glBindVertexArray(sphere_vao);
    glGenBuffers(1, &sphere_vbo);
    glGenBuffers(1, &sphere_ebo);
    GLuint sphere_index_count;
    {
        auto [vertices, indices] = generate_sphere(1.f, 16);

        glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        sphere_index_count = indices.size();
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, tangent));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, texcoords));

    auto watch_tower_vertex_shader = create_shader(GL_VERTEX_SHADER, watch_tower_vertex_shader_source);
    auto watch_tower_fragment_shader = create_shader(GL_FRAGMENT_SHADER, watch_tower_fragment_shader_source);
    auto watch_tower_program = create_program(watch_tower_vertex_shader, watch_tower_fragment_shader);

    GLuint watch_tower_model_location = glGetUniformLocation(program, "model");
    GLuint watch_tower_view_location = glGetUniformLocation(program, "view");
    GLuint watch_tower_projection_location = glGetUniformLocation(program, "projection");
    GLuint watch_tower_transform_location = glGetUniformLocation(program, "transform");
    GLuint watch_tower_ambient_location = glGetUniformLocation(program, "ambient");
    GLuint watch_tower_light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint watch_tower_light_color_location = glGetUniformLocation(program, "light_color");
    GLuint watch_tower_camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint watch_tower_shadow_map_location = glGetUniformLocation(program, "shadow_map");
    GLuint watch_tower_texture_location = glGetUniformLocation(program, "texture");
    GLuint watch_tower_normal_location = glGetUniformLocation(program, "normal_texture");

    std::string project_root = PROJECT_ROOT;
    GLuint snow_texture = load_texture(project_root + "/textures/snow_texture.jpeg");
    GLuint environment_texture = load_texture(project_root + "/textures/environment_map.jpg");
    GLuint watch_tower_texture = load_texture(project_root + "/textures/Wood_Tower_Col.jpg");
    GLuint watch_tower_normal_texture = load_texture(project_root + "/textures/Wood_Tower_Nor.jpg");

    std::string watch_tower_path = project_root + "/textures/watch_tower.obj";
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> watch_tower_shapes;
    std::vector<tinyobj::material_t> watch_tower_materials;
    tinyobj::LoadObj(&attrib, &watch_tower_shapes, &watch_tower_materials, nullptr, watch_tower_path.c_str(), watch_tower_path.c_str());

    obj_data watch_tower_data;
    {
        // Loop over shapes
        for (size_t s = 0; s < watch_tower_shapes.size(); s++)
        {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < watch_tower_shapes[s].mesh.num_face_vertices.size(); f++)
            {
                size_t fv = size_t(watch_tower_shapes[s].mesh.num_face_vertices[f]);

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++)
                {
                    watch_tower_data.vertices.push_back(obj_data::vertex());

                    // access to vertex
                    tinyobj::index_t idx = watch_tower_shapes[s].mesh.indices[index_offset + v];

                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                    watch_tower_data.vertices.back().position = {vx, vy, vz};

                    // Check if `normal_index` is zero or positive. negative = no normal data
                    if (idx.normal_index >= 0)
                    {
                        tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                        tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                        tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                        watch_tower_data.vertices.back().normal = {nx, ny, nz};
                    }

                    // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                    if (idx.texcoord_index >= 0)
                    {
                        tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                        tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                        watch_tower_data.vertices.back().texcoord = {tx, ty};
                    }
                    // Optional: vertex colors
                    // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                    // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                    // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
                }
                index_offset += fv;

                // per-face material
                watch_tower_shapes[s].mesh.material_ids[f];
            }
        }
    }
    GLuint watch_tower_vao, watch_tower_vbo;
    glGenVertexArrays(1, &watch_tower_vao);
    glBindVertexArray(watch_tower_vao);

    glGenBuffers(1, &watch_tower_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, watch_tower_vbo);
    glBufferData(GL_ARRAY_BUFFER, watch_tower_data.vertices.size() * sizeof(obj_data::vertex), watch_tower_data.vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *)(12));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void *)(24));

    float infty = std::numeric_limits<float>::infinity();
    float min_x = infty, max_x = -infty;
    float min_y = infty, max_y = -infty;
    float min_z = infty, max_z = -infty;
    for (obj_data::vertex el : watch_tower_data.vertices)
    {
        min_x = std::min(min_x, el.position[0]);
        max_x = std::max(max_x, el.position[0]);

        min_y = std::min(min_y, el.position[1]);
        max_y = std::max(max_y, el.position[1]);

        min_z = std::min(min_z, el.position[2]);
        max_z = std::max(max_z, el.position[2]);
    }

    std::vector<std::vector<float>> watch_tower_bounding_box(8, std::vector<float>(3));
    watch_tower_bounding_box = {
        {min_x, min_y, min_z},
        {min_x, min_y, max_z},
        {min_x, max_y, min_z},
        {min_x, max_y, max_z},
        {max_x, min_y, min_z},
        {max_x, min_y, max_z},
        {max_x, max_y, min_z},
        {max_x, max_y, max_z},
    };

    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    bool paused = false;
    std::map<SDL_Keycode, bool> button_down;
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 direction;
    float yaw = -90.f, pitch = 0.f;
    const float cameraMovementSpeed = 0.05f;
    const float cameraRotationSpeed = 50.f;

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
        time += !paused * dt;

        if (button_down[SDLK_LEFT])
            yaw -= cameraRotationSpeed * dt;
        if (button_down[SDLK_RIGHT])
            yaw += cameraRotationSpeed * dt;
        if (button_down[SDLK_UP])
            pitch += cameraRotationSpeed * dt;
        if (button_down[SDLK_DOWN])
            pitch -= cameraRotationSpeed * dt;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(direction);

        if (button_down[SDLK_a])
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraMovementSpeed;
        if (button_down[SDLK_d])
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraMovementSpeed;
        if (button_down[SDLK_w])
            cameraPos += cameraMovementSpeed * cameraFront;
        if (button_down[SDLK_s])
            cameraPos -= cameraMovementSpeed * cameraFront;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float near = 0.01f;
        float far = 100.f;
        float top = near;
        float right = (top * width) / height;

        glm::mat4 model = glm::mat4(1.f);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 light_direction = glm::normalize(glm::vec3(std::sin(time * 0.5f) * 3, 2.f, std::cos(time * 0.5f) * 3));

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 light_z = -light_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, {0.f, 1.f, 0.f}));
        glm::vec3 light_y = glm::cross(light_x, light_z);
        float c_x = (max_x + min_x) / 2;
        float c_y = (max_y + min_y) / 2;
        float c_z = (max_z + min_z) / 2;
        float light_x_mx = -infty;
        for (auto &el : watch_tower_bounding_box)
        {
            light_x_mx = std::max(light_x_mx, std::abs(glm::dot({el[0] - c_x, el[1] - c_y, el[2] - c_z}, light_x)));
        }
        float light_y_mx = -infty;
        for (auto &el : watch_tower_bounding_box)
        {
            light_y_mx = std::max(light_y_mx, std::abs(glm::dot({el[0] - c_x, el[1] - c_y, el[2] - c_z}, light_y)));
        }
        float light_z_mx = -infty;
        for (auto &el : watch_tower_bounding_box)
        {
            light_z_mx = std::max(light_z_mx, std::abs(glm::dot({el[0] - c_x, el[1] - c_y, el[2] - c_z}, light_z)));
        }
        light_x *= light_x_mx;
        light_y *= light_y_mx;
        light_z *= light_z_mx;
        glm::mat4 transform = glm::mat4(1.f);
        transform = {
            {light_x[0], light_y[0], light_z[0], c_x},
            {light_x[1], light_y[1], light_z[1], c_y},
            {light_x[2], light_y[2], light_z[2], c_z},
            {0.0, 0.0, 0.0, 1.0}};
        transform = glm::inverse(glm::transpose(transform));

        {
            glUseProgram(background_program);

            glm::mat4 view_projection_inverse = glm::inverse(projection * view);
            glUniformMatrix4fv(view_projection_inverse_location, 1, GL_FALSE, reinterpret_cast<float *>(&view_projection_inverse));
            glUniform3fv(background_camera_position_location, 1, reinterpret_cast<float *>(&camera_position));

            glUniform1i(background_environment_texture_location, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, environment_texture);

            glBindVertexArray(background_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);

            model = glm::scale(model, glm::vec3(0.09f));
            model = glm::translate(model, glm::vec3(0.f, -3.f, 0.f));

            glUseProgram(watch_tower_program);
            glUniformMatrix4fv(watch_tower_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
            glUniformMatrix4fv(watch_tower_view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
            glUniformMatrix4fv(watch_tower_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
            glUniformMatrix4fv(watch_tower_transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));
            glUniform3fv(watch_tower_light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
            glUniform3fv(watch_tower_camera_position_location, 1, reinterpret_cast<float *>(&cameraPos));
            glUniform3f(watch_tower_light_color_location, 0.8f, 0.8f, 0.8f);
            glUniform3f(watch_tower_ambient_location, 0.2f, 0.2f, 0.2f);
            glUniform1i(watch_tower_shadow_map_location, 1);

            glUniform1i(watch_tower_texture_location, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, watch_tower_texture);

            glUniform1i(watch_tower_normal_location, 1);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, watch_tower_normal_texture);

            glBindVertexArray(watch_tower_vao);
            glDrawArrays(GL_TRIANGLES, 0, watch_tower_shapes[0].mesh.indices.size());

            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);

            model = glm::mat4(1.f);
        }

        {
            glEnable(GL_DEPTH_TEST);

            glUseProgram(floor_program);
            glUniformMatrix4fv(floor_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
            glUniformMatrix4fv(floor_view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
            glUniformMatrix4fv(floor_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));

            glUniform1i(floor_snow_texture_location, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, snow_texture);

            glBindVertexArray(floor_vao);
            glDrawElements(GL_TRIANGLES, half_sphere_index_count, GL_UNSIGNED_INT, nullptr);

            glDisable(GL_DEPTH_TEST);
        }

        {
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);

            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glUseProgram(program);
            glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
            glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
            glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
            glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
            glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));

            glUniform1i(snow_texture_location, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, snow_texture);

            glUniform1i(environment_texture_location, 1);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, environment_texture);

            glBindVertexArray(sphere_vao);
            glDrawElements(GL_TRIANGLES, sphere_index_count, GL_UNSIGNED_INT, nullptr);

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
        }

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
