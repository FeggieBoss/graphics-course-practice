#include <SDL2/SDL.h>
#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

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

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;

layout (location = 0) in vec2 in_position;
layout (location = 1) in vec4 in_color;
out vec4 color;
void main()
{
    gl_Position = view * vec4(in_position, 0.0, 1.0);
    color = in_color;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core
in vec4 color;
layout (location = 0) out vec4 out_color;
void main()
{
    out_color = color;
}
)";

struct vec2
{
    float x;
    float y;
};

struct cords {
    int x;
    int y;
};

struct color {
    std::uint8_t color[4];
};

struct vertex {
    vec2 position;
    color col;
};

struct segment {
    vec2 a, b;
};

struct point {
    cords pos;
    cords mov_vec;
    int r;

    point(int width, int height) {
        pos = { get_rng()%width+1, get_rng()%height+1 };
        mov_vec = { (get_rng()%2?1:-1), (get_rng()%2?1:-1) };
        r = get_rng()%20 + 10;
    }

    static int get_rng() {
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        return std::abs(int(rng()));
    }
};

std::uint8_t foo(int x, int y, std::vector<point> &pnts) {
    float ret = 0;
    for (auto &el : pnts) {
        float dist = (x-el.pos.x)*(x-el.pos.x) + (y-el.pos.y)*(y-el.pos.y);
        dist /= el.r*el.r;
        dist /= 5;
        ret += 100 * exp(-dist);
    }   
    if(ret>0.01) {
        if(ret<0.5)
            ret*=100;
        else if(ret<10)
            ret*=30;
        else if(ret<50)
            ret*=19;
        else
            ret*=5;
    }

    return std::min(ret,255.f);
}

std::vector<segment> get_segment(float x, float y, std::vector<point> &pnts, int bound) {
    int mask = 0;

    if(int(foo(x,y,pnts)) >= bound) {
        mask += 8;
    }
    x += 10;
    if(int(foo(x,y,pnts)) >= bound) {
        mask += 4;
    }
    y += 10;
    if(int(foo(x,y,pnts)) >= bound) {
        mask += 2;
    }
    x -= 10;
    if(int(foo(x,y,pnts)) >= bound) {
        mask += 1;
    }
    y -= 10;

    switch (mask)
    {
    case 0:
        return {};
    case 1:
        return {{{x, y+5}, {x+5, y+10}}};
    case 2:
        return {{{x+5, y+10}, {x+10, y+5}}};
    case 3:
        return {{{x, y+5}, {x+10, y+5}}};
    case 4:
        return {{{x+5, y}, {x+10, y+5}}};
    case 5:
        return {{{x, y+5}, {x+5, y}}, {{x+5, y+10}, {x+10, y+5}}};
    case 6:
        return {{{x+5, y}, {x+5, y+10}}};
    case 7:
        return {{{x, y+5}, {x+5, y}}};
    case 8:
        return {{{x, y+5}, {x+5, y}}};
    case 9:
        return {{{x+5, y}, {x+5, y+10}}};
    case 10:
        return {{{x, y+5}, {x+5, y+10}}, {{x+5, y}, {x+10, y+5}}};
    case 11:
        return {{{x+5, y}, {x+10, y+5}}};
    case 12:
        return {{{x, y+5}, {x+10, y+5}}};
    case 13:
        return {{{x+5, y+10}, {x+10, y+5}}};
    case 14:
        return {{{x, y+5}, {x+5, y+10}}};
    case 15:
        return {};
    };
    return {};
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
    
    std::vector<vertex> vertices_iso; // для изолиний

    std::vector<vec2> vertices; // ({
    //     {{0,0},{255,0,0,1}},
    //     {{0,float(height)},{0,255,0,1}},
    //     {{float(width),0},{0,0,255,1}}
    // });

    for(int i=0;i<=width;i+=10) {
        for(int j=0;j<=height;j+=10) {
            vertices.push_back({float(i), float(j)});
        }
    }

    std::vector<std::uint32_t> indices;
    for(int i=0;i<=width-10; i+=10) {
        for(int j=0;j<=height-10; j+=10) {
            indices.push_back((height/10+1)*(i/10) + (j/10));
            indices.push_back((height/10+1)*(i/10) + (j/10) + 1);
            indices.push_back((height/10+1)*(i/10+1) + (j/10));

            indices.push_back((height/10+1)*(i/10+1) + (j/10));
            indices.push_back((height/10+1)*(i/10+1) + (j/10) + 1);
            indices.push_back((height/10+1)*(i/10) + (j/10) + 1);
        }
    }

    std::vector<color> colors(vertices.size());

    GLuint view_location = glGetUniformLocation(program, "view");

    //VAO для сетки
    GLuint vao;
    glGenVertexArrays(1, &vao);

    //VAO для изолиний
    GLuint vao_iso;
    glGenVertexArrays(1, &vao_iso);

    //VBO для координат
    GLuint vbo_pos;
    glGenBuffers(1, &vbo_pos);

    //VBO для изолиний
    GLuint vbo_iso;
    glGenBuffers(1, &vbo_iso);

    //VBO для цветов
    GLuint vbo_color;
    glGenBuffers(1, &vbo_color);

    // настройка для изолиний
    glBindVertexArray(vao_iso);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_iso);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(2*4));
    //glBufferData(GL_ARRAY_BUFFER, vertices_iso.size() * sizeof(vertex), vertices_iso.data(), GL_STREAM_DRAW);

    // настройка для координат
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*)(0));
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), vertices.data(), GL_STREAM_DRAW);

    // настройка для цветов
    glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(color), (void*)(0));

    //EBO
    GLuint ebo;
    glGenBuffers (1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(std::uint32_t), indices.data(), GL_STATIC_DRAW);

    // блуждающие точки
    std::vector<point> pnts(10, point(width,height));
    for(int i=0;i<10;++i)
        pnts[i]=point(width,height);
    int old_width = width;
    int old_height = height;

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
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        // буфер цветов зачистить, когда рисуем новое
        glClear(GL_COLOR_BUFFER_BIT);

        // блуждание
        for (auto &el : pnts) {
            if(el.pos.x == 0 || el.pos.x == old_width)
                el.mov_vec.x *= -1;
            if(el.pos.y == 0 || el.pos.y == old_height)
                el.mov_vec.y *= -1;

            el.pos.x += el.mov_vec.x;
            el.pos.y += el.mov_vec.y;
        }


        // трансформируем координаты в дробные
        float view[16] =
        {
            2.0f/width, 0.f, 0.f, -1.f,
            0.f, -2.0f/height, 0.f, 1.0f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };

        // зарядил шейдеры
        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);


        // задаю новые цвета
        for(int i=0;i<colors.size();++i) {
            colors[i] = {foo(vertices[i].x, vertices[i].y, pnts),0,255,1};
            //if(int(colors[i].color[0])>0)
            //    colors[i] = {colors[i].color[0], 0, 0, 1};
        }

        // рисую
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(color), colors.data(), GL_STREAM_DRAW);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);    

        vertices_iso.clear();
        for(int i=0;i<vertices.size(); ++i) {
            if(vertices[i].x == old_width || vertices[i].y == old_height)
                continue;

            auto g = get_segment(vertices[i].x, vertices[i].y, pnts, 30);
            colors[i].color[1] = 150;
            for(auto &el : g) {
                vertices_iso.push_back({el.a, colors[i]});
                vertices_iso.push_back({el.b, colors[i]});
            }

            g = get_segment(vertices[i].x, vertices[i].y, pnts, 200);
            colors[i].color[1] = 255;
            for(auto &el : g) {
                vertices_iso.push_back({el.a, colors[i]});
                vertices_iso.push_back({el.b, colors[i]});
            }
        }

        glBindVertexArray(vao_iso);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_iso);
        glBufferData(GL_ARRAY_BUFFER, vertices_iso.size() * sizeof(vertex), vertices_iso.data(), GL_STREAM_DRAW);
        glLineWidth(10); 
        glDrawArrays(GL_LINES, 0, vertices_iso.size());

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
