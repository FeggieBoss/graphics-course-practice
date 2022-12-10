#include <SDL2/SDL.h>
#include <GL/glew.h>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "obj_parser.hpp"

#include "stb_image.h"

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cassert>
#include <map>

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

const char fragment_shader_source[] =
R"(#version 330 core
uniform float glossiness;
uniform float shininess;

uniform vec3 ambient;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 camera_position;

uniform mat4 transform;
uniform sampler2D shadow_map;
uniform sampler2D texture;

uniform sampler2D map_d;
uniform int flag_map_d;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

float diffuse(vec3 direction) {
    return max(0.0, dot(normal, direction));
}
float specular(vec3 direction) {
    vec3 reflected_direction = 2.0 * normal * dot(normal, direction) - direction;
    vec3 view_direction = normalize(camera_position - position);
    return pow(max(0.0, dot(reflected_direction, view_direction)), shininess) * glossiness;
}
float phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

void main()
{
    if(flag_map_d == 1 && texture2D(map_d, texcoord).x < 0.5) discard;

    vec4 shadow_pos = transform * vec4(position, 1.0);
    shadow_pos /= shadow_pos.w;
    shadow_pos = shadow_pos * 0.5 + vec4(0.5);
    
    bool in_shadow_texture = (shadow_pos.x > 0.0) && (shadow_pos.x < 1.0) && (shadow_pos.y > 0.0) && (shadow_pos.y < 1.0) && (shadow_pos.z > 0.0) && (shadow_pos.z < 1.0);
    float shadow_factor = 1.0;
    
    float factor = 1.0;
    if (in_shadow_texture) {
        vec2 sum = vec2(0.0);
        float sum_w = 0.0;
        const int N = 2;
        float radius = 3.0;
        for (int x = -N; x <= N; ++x) {
            for (int y = -N; y <= N; ++y) {
                float c = exp(-float(x*x + y*y) / (radius*radius));
                sum += c * texture2D(shadow_map, shadow_pos.xy + vec2(x,y) / vec2(textureSize(shadow_map, 0))).xy;
                sum_w += c;
            }
        }


        vec2 data = sum / sum_w;
        float bias = -0.005;
        float mu = data.r;
        float sigma = data.g - mu * mu;
        float z = shadow_pos.z + bias;
        factor = (z < mu) ? 1.0 : sigma / (sigma + (z - mu) * (z - mu));
        
        float delta = 0.125;
        if(factor<delta) {
            factor = 0;
        }
        else {
            factor = (factor-delta) * 1.f/(1-delta);
        }
    }
    vec3 light = ambient;
    light += light_color * phong(light_direction) * factor;
    vec3 color = texture2D(texture, texcoord).xyz * light;
    out_color = vec4(color, 1.0);
}
)";


const char shadow_vertex_shader_source[] =
R"(#version 330 core
uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;
layout (location = 2) in vec2 in_texcoord;

out vec2 texcoord;

void main()
{
    gl_Position = transform * model * vec4(in_position, 1.0);
    texcoord = vec2(in_texcoord.x, 1.f-in_texcoord.y);
}
)";

const char shadow_fragment_shader_source[] =
R"(#version 330 core
uniform sampler2D map_d;

in vec4 gl_FragCoord;
in vec2 texcoord;

uniform int flag_map_d;

layout (location = 0) out vec4 z_zz;

void main()
{   
    if(flag_map_d == 1 && texture2D(map_d, texcoord).x < 0.5) discard;

    float z = gl_FragCoord.z;
    z_zz = vec4(z, z * z + 0.25 * (dFdx(z)*dFdx(z) + dFdy(z)*dFdy(z)), 0.0, 0.0);
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

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint transform_location = glGetUniformLocation(program, "transform");
    GLuint ambient_location = glGetUniformLocation(program, "ambient");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint light_color_location = glGetUniformLocation(program, "light_color");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");
    GLuint texture_location = glGetUniformLocation(program, "texture");
    GLuint glossiness_location = glGetUniformLocation(program, "glossiness");
    GLuint shininess_location = glGetUniformLocation(program, "shininess");
    GLuint map_d_location = glGetUniformLocation(program, "map_d");
    GLuint flag_map_d_location = glGetUniformLocation(program, "flag_map_d");















    auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
    auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
    auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

    GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");
    GLuint shadow_transform_location = glGetUniformLocation(shadow_program, "transform");
    GLuint shadow_map_d_location = glGetUniformLocation(shadow_program, "map_d");
    GLuint shadow_flag_map_d_location = glGetUniformLocation(shadow_program, "flag_map_d");


    













    std::string project_root = PROJECT_ROOT;
    std::string scene_path = project_root + "/sponza.obj";
    std::string materials_dir = project_root + "/";

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, nullptr, scene_path.c_str(), materials_dir.c_str());

    obj_data scene;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);


            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                scene.vertices.push_back(obj_data::vertex());

                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
                scene.vertices.back().position = {vx, vy, vz};

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];

                    scene.vertices.back().normal = {nx, ny, nz};
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                    tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];

                    scene.vertices.back().texcoord = {tx, ty};
                }
                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
    























    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, scene.vertices.size() * sizeof(obj_data::vertex), scene.vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(12));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(24));


















    GLsizei shadow_map_resolution = 1024;
    GLuint shadow_map;
    glGenTextures(1, &shadow_map);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, shadow_map_resolution, shadow_map_resolution, 0, GL_RGBA, GL_FLOAT, NULL);
    
    GLuint shadow_fbo;
    glGenFramebuffers(1, &shadow_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadow_map, 0);
    
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, shadow_map_resolution, shadow_map_resolution);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Incomplete framebuffer!");
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
































    int last_textrue_location_id = 1; // 0/1 will be free
    std::map<std::string, int> texture_pos;

    glUseProgram(program);
    for (size_t s = 0; s < shapes.size(); s++) {
        int id = shapes[s].mesh.material_ids[0];
        std::string ambient_texname = materials[id].ambient_texname;

        if(texture_pos.find(ambient_texname)!=texture_pos.end()) continue;
        texture_pos[ambient_texname] = ++last_textrue_location_id;

        GLuint textureID=0;
        glGenTextures(1, &textureID);
        glActiveTexture(GL_TEXTURE0 + last_textrue_location_id);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        int texture_width, texture_height, texture_nrChannels;
        
        for(char &c : ambient_texname) if(c=='\\') c='/';
        std::string path = project_root + "/" + ambient_texname;
        unsigned char* pixels_texture = stbi_load(path.c_str(), &texture_width, &texture_height, &texture_nrChannels, 4);
                    
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_texture);

        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(pixels_texture);        
    }


    std::map<std::string, int> map_d_pos;
    for (size_t s = 0; s < shapes.size(); s++) {
        int id = shapes[s].mesh.material_ids[0];
        std::string alpha_texname = materials[id].alpha_texname;

        if(map_d_pos.find(alpha_texname)!=map_d_pos.end()) continue;
        map_d_pos[alpha_texname] = ++last_textrue_location_id;

        GLuint textureID=0;
        glGenTextures(1, &textureID);
        glActiveTexture(GL_TEXTURE0 + last_textrue_location_id);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        int texture_width, texture_height, texture_nrChannels;
        
        for(char &c : alpha_texname) if(c=='\\') c='/';
        std::string path = project_root + "/" + alpha_texname;
        unsigned char* pixels_texture = stbi_load(path.c_str(), &texture_width, &texture_height, &texture_nrChannels, 4);
                    
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels_texture);

        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(pixels_texture);        
    }



















    float infty = std::numeric_limits<float>::infinity();
    float min_x = infty, max_x = -infty;
    float min_y = infty, max_y = -infty;
    float min_z = infty, max_z = -infty;
    for(obj_data::vertex el : scene.vertices) {
        min_x = std::min(min_x, el.position[0]);
        max_x = std::max(max_x, el.position[0]);

        min_y = std::min(min_y, el.position[1]);
        max_y = std::max(max_y, el.position[1]);

        min_z = std::min(min_z, el.position[2]);
        max_z = std::max(max_z, el.position[2]);
    }

    std::vector<std::vector<float>> v(8, std::vector<float>(3));
    v = {
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
    glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  3.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    glm::vec3 direction;
    float yaw = -90.f, pitch = 0.f;
    const float cameraMovementSpeed = 10.f;
    const float cameraRotationSpeed = 50.f;
















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
        if (!paused)
            time += dt;

        if (button_down[SDLK_LEFT])
            yaw -= cameraRotationSpeed * dt;
        if (button_down[SDLK_RIGHT])
            yaw += cameraRotationSpeed * dt;
        if (button_down[SDLK_UP])
            pitch += cameraRotationSpeed * dt;
        if (button_down[SDLK_DOWN])
            pitch -= cameraRotationSpeed * dt;

        if(pitch > 89.0f)
            ` =  89.0f;
        if(pitch < -89.0f)
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
















        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, shadow_map_resolution, shadow_map_resolution);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);



        glm::mat4 model(1.f);
        glm::vec3 light_direction = glm::normalize(glm::vec3(std::cos(time * 0.5f), 1.f, std::sin(time * 0.5f)));
        glm::vec3 light_z = -light_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, {0.f, 1.f, 0.f}));
        glm::vec3 light_y = glm::cross(light_x, light_z);
        float c_x = (max_x + min_x) / 2;
        float c_y = (max_y + min_y) / 2;
        float c_z = (max_z + min_z) / 2;
        float light_x_mx = -infty;
        for(auto &el : v) {
            light_x_mx = std::max(light_x_mx, std::abs(glm::dot({el[0]-c_x,el[1]-c_y,el[2]-c_z}, light_x)));
        }
        float light_y_mx = -infty;
        for(auto &el : v) {
            light_y_mx = std::max(light_y_mx, std::abs(glm::dot({el[0]-c_x,el[1]-c_y,el[2]-c_z}, light_y)));
        }
        float light_z_mx = -infty;
        for(auto &el : v) {
            light_z_mx = std::max(light_z_mx, std::abs(glm::dot({el[0]-c_x,el[1]-c_y,el[2]-c_z}, light_z)));
        }
        light_x *= light_x_mx;
        light_y *= light_y_mx;
        light_z *= light_z_mx;
        glm::mat4 transform = glm::mat4(1.f);
        transform = {
            {light_x[0], light_y[0], light_z[0], c_x},
            {light_x[1], light_y[1], light_z[1], c_y},
            {light_x[2], light_y[2], light_z[2], c_z},
            {0.0, 0.0, 0.0, 1.0}
        };
        transform = glm::inverse(glm::transpose(transform));



        glUseProgram(shadow_program);
        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(shadow_transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));
        
        int first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            int id = shapes[s].mesh.material_ids[0];
            std::string alpha_texname = materials[id].alpha_texname;

            glUniform1i(shadow_map_d_location, map_d_pos[alpha_texname]);
            glUniform1i(shadow_flag_map_d_location, !alpha_texname.empty());

            glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
            first += shapes[s].mesh.indices.size();
        }




        

        




















        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glClearColor(0.8f, 0.8f, 0.9f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        


        float near = 0.09f;
        float far = 3500.f;
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);
        


        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&cameraPos));
        glUniform3f(light_color_location, 0.8f, 0.8f, 0.8f);
        glUniform3f(ambient_location, 0.2f, 0.2f, 0.2f);
        glUniform1i(shadow_map_location, 1);















        first = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            int id = shapes[s].mesh.material_ids[0];
            std::string ambient_texname = materials[id].ambient_texname;
            std::string alpha_texname = materials[id].alpha_texname;

            glUniform1i(map_d_location, map_d_pos[alpha_texname]);
            glUniform1i(flag_map_d_location, !alpha_texname.empty());

            glUniform1i(texture_location, texture_pos[ambient_texname]);
            glUniform1f(glossiness_location, materials[id].specular[0]);
            glUniform1f(shininess_location, materials[id].shininess);

            glDrawArrays(GL_TRIANGLES, first, (GLint)shapes[s].mesh.indices.size());
            first += shapes[s].mesh.indices.size();
        }






        
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
