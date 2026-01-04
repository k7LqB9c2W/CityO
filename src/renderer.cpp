#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "renderer.h"

#include "mesh_cache.h"

#include <SDL.h>
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <windows.h>
#include <wincodec.h>

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

GLuint MakeProgram(const char* vsSrc, const char* fsSrc);
bool GLCheckShader(GLuint shader, const char* label);
bool GLCheckProgram(GLuint prog);
void SetupInstanceAttribs(GLuint vao, GLuint instanceVbo);
void UploadDynamicVerts(GLuint vbo, std::size_t& capacityBytes, const std::vector<glm::vec3>& verts);
void UploadDynamicMats(GLuint vbo, std::size_t& capacityBytes, const std::vector<glm::mat4>& mats);
std::wstring Utf8ToWide(const char* str);
bool LoadImageWIC(const char* path, std::vector<uint8_t>& outPixels, int& outW, int& outH);
GLuint CreateTextureFromRGBA(const uint8_t* pixels, int w, int h);
GLuint CreateSolidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
GLuint LoadTexture2D(const char* path, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool* outOk);
GLuint CreateSolidCubemap(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
GLuint LoadCubemap(const char* faces[6], bool* outOk);

bool GLCheckShader(GLuint shader, const char* label) {
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string log((size_t)len, '\0');
        glGetShaderInfoLog(shader, len, &len, log.data());
        SDL_Log("Shader compile failed (%s): %s", label, log.c_str());
        return false;
    }
    return true;
}

bool GLCheckProgram(GLuint prog) {
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log((size_t)len, '\0');
        glGetProgramInfoLog(prog, len, &len, log.data());
        SDL_Log("Program link failed: %s", log.c_str());
        return false;
    }
    return true;
}

GLuint MakeProgram(const char* vsSrc, const char* fsSrc) {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSrc, nullptr);
    glCompileShader(vs);
    if (!GLCheckShader(vs, "VS")) {
        glDeleteShader(vs);
        return 0;
    }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, nullptr);
    glCompileShader(fs);
    if (!GLCheckShader(fs, "FS")) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    if (!GLCheckProgram(prog)) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        glDeleteProgram(prog);
        return 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void SetupInstanceAttribs(GLuint vao, GLuint instanceVbo) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(HouseInstanceGPU), (void*)0);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(HouseInstanceGPU), (void*)(sizeof(glm::vec4)));

    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);

    glBindVertexArray(0);
}

void UploadDynamicVerts(GLuint vbo, std::size_t& capacityBytes, const std::vector<glm::vec3>& verts) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    std::size_t bytes = verts.size() * sizeof(glm::vec3);
    if (bytes == 0) {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
        capacityBytes = 0;
    } else {
        if (bytes > capacityBytes) {
            capacityBytes = (std::size_t)(bytes * 1.5f) + 256;
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)capacityBytes, nullptr, GL_DYNAMIC_DRAW);
        } else {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)capacityBytes, nullptr, GL_DYNAMIC_DRAW); // orphan
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)bytes, verts.data());
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void UploadDynamicMats(GLuint vbo, std::size_t& capacityBytes, const std::vector<glm::mat4>& mats) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    std::size_t bytes = mats.size() * sizeof(glm::mat4);
    if (bytes == 0) {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
        capacityBytes = 0;
    } else {
        if (bytes > capacityBytes) {
            capacityBytes = (std::size_t)(bytes * 1.5f) + 256;
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)capacityBytes, nullptr, GL_DYNAMIC_DRAW);
        } else {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)capacityBytes, nullptr, GL_DYNAMIC_DRAW); // orphan
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)bytes, mats.data());
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

std::wstring Utf8ToWide(const char* str) {
    if (!str || !str[0]) return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
    if (len <= 1) return {};
    std::wstring out(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str, -1, out.data(), len);
    out.pop_back(); // remove null terminator
    return out;
}

bool LoadImageWIC(const char* path, std::vector<uint8_t>& outPixels, int& outW, int& outH) {
    outPixels.clear();
    outW = 0;
    outH = 0;
    if (!path || !path[0]) return false;

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool coInit = (hr == S_OK || hr == S_FALSE);
    if (hr == RPC_E_CHANGED_MODE) {
        coInit = false;
    } else if (FAILED(hr)) {
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IWICImagingFactory, (void**)&factory);
    if (FAILED(hr)) {
        if (coInit) CoUninitialize();
        return false;
    }

    std::wstring wpath = Utf8ToWide(path);
    if (wpath.empty()) {
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    IWICBitmapDecoder* decoder = nullptr;
    hr = factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) {
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) {
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    IWICFormatConverter* converter = nullptr;
    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr)) {
        frame->Release();
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    hr = converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA,
                               WICBitmapDitherTypeNone, nullptr, 0.0,
                               WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        converter->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    UINT w = 0;
    UINT h = 0;
    converter->GetSize(&w, &h);
    if (w == 0 || h == 0) {
        converter->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    outPixels.resize((size_t)w * (size_t)h * 4);
    hr = converter->CopyPixels(nullptr, w * 4, (UINT)outPixels.size(), outPixels.data());

    converter->Release();
    frame->Release();
    decoder->Release();
    factory->Release();
    if (coInit) CoUninitialize();

    if (FAILED(hr)) {
        outPixels.clear();
        return false;
    }

    outW = (int)w;
    outH = (int)h;
    return true;
}

GLuint CreateTextureFromRGBA(const uint8_t* pixels, int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
#ifdef GL_TEXTURE_MAX_ANISOTROPY_EXT
    float maxAniso = 0.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
    if (maxAniso > 0.0f) {
        float aniso = (maxAniso < 8.0f) ? maxAniso : 8.0f;
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
    }
#endif
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

GLuint CreateSolidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint8_t pixel[4] = { r, g, b, a };
    return CreateTextureFromRGBA(pixel, 1, 1);
}

GLuint LoadTexture2D(const char* path, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool* outOk) {
    std::vector<uint8_t> pixels;
    int w = 0;
    int h = 0;
    if (LoadImageWIC(path, pixels, w, h)) {
        if (outOk) *outOk = true;
        return CreateTextureFromRGBA(pixels.data(), w, h);
    }
    if (outOk) *outOk = false;
    SDL_Log("Texture load failed: %s", path);
    return CreateSolidTexture(r, g, b, a);
}

GLuint CreateSolidCubemap(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    uint8_t pixel[4] = { r, g, b, a };
    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return tex;
}

GLuint LoadCubemap(const char* faces[6], bool* outOk) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    bool ok = true;
    int w = 0;
    int h = 0;
    std::vector<uint8_t> pixels;
    for (int i = 0; i < 6; ++i) {
        int wi = 0;
        int hi = 0;
        if (!LoadImageWIC(faces[i], pixels, wi, hi)) {
            ok = false;
            break;
        }
        if (i == 0) {
            w = wi;
            h = hi;
        } else if (wi != w || hi != h) {
            ok = false;
            break;
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    if (!ok) {
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glDeleteTextures(1, &tex);
        if (outOk) *outOk = false;
        return CreateSolidCubemap(120, 160, 210, 255);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    if (outOk) *outOk = true;
    return tex;
}

} // namespace

bool Renderer::init() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    const char* vsBasic = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uViewProj;
        uniform mat4 uModel;
        void main() {
            gl_Position = uViewProj * uModel * vec4(aPos, 1.0);
        }
    )";

    const char* vsInstanced = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=2) in vec4 iPosYaw;   // xyz, yaw
        layout(location=3) in vec4 iScaleVar; // xyz scale
        uniform mat4 uViewProj;
        void main() {
            float yaw = iPosYaw.w;
            mat3 R = mat3(
                cos(yaw), 0.0, -sin(yaw),
                0.0,      1.0,  0.0,
                sin(yaw), 0.0,  cos(yaw)
            );
            vec3 scaled = R * (aPos * iScaleVar.xyz);
            vec4 world = vec4(iPosYaw.xyz + scaled, 1.0);
            world.y += 0.05; // lift houses off the ground to avoid z-fighting
            gl_Position = uViewProj * world;
        }
    )";

    const char* vsGround = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uViewProj;
        uniform mat4 uModel;
        uniform float uGrassTileM;
        uniform float uNoiseTileM;
        out vec2 vGrassUV;
        out vec2 vNoiseUV;
        void main() {
            vec4 world = uModel * vec4(aPos, 1.0);
            vGrassUV = world.xz / uGrassTileM;
            vNoiseUV = world.xz / uNoiseTileM;
            gl_Position = uViewProj * world;
        }
    )";

    const char* vsSky = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        out vec3 vDir;
        uniform mat4 uViewProj;
        void main() {
            vDir = aPos;
            vec4 pos = uViewProj * vec4(aPos, 1.0);
            gl_Position = pos.xyww;
        }
    )";

    const char* fsColor = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 uColor;
        uniform float uAlpha;
        void main() {
            FragColor = vec4(uColor, uAlpha);
        }
    )";

    const char* fsSky = R"(
        #version 330 core
        in vec3 vDir;
        out vec4 FragColor;
        uniform samplerCube uSkybox;
        void main() {
            FragColor = texture(uSkybox, normalize(vDir));
        }
    )";

    const char* fsGround = R"(
        #version 330 core
        in vec2 vGrassUV;
        in vec2 vNoiseUV;
        out vec4 FragColor;
        uniform sampler2D uGrassTex;
        uniform sampler2D uNoiseTex;
        void main() {
            vec3 grass = texture(uGrassTex, vGrassUV).rgb;
            float n = texture(uNoiseTex, vNoiseUV).r;
            float shade = mix(0.85, 1.15, n);
            vec3 color = grass * shade;
            FragColor = vec4(color, 1.0);
        }
    )";

    progBasic = MakeProgram(vsBasic, fsColor);
    progInst = MakeProgram(vsInstanced, fsColor);
    progGround = MakeProgram(vsGround, fsGround);
    progSky = MakeProgram(vsSky, fsSky);
    if (!progBasic || !progInst || !progGround || !progSky) return false;

    locVP_B = glGetUniformLocation(progBasic, "uViewProj");
    locM_B = glGetUniformLocation(progBasic, "uModel");
    locC_B = glGetUniformLocation(progBasic, "uColor");
    locA_B = glGetUniformLocation(progBasic, "uAlpha");

    locVP_I = glGetUniformLocation(progInst, "uViewProj");
    locC_I = glGetUniformLocation(progInst, "uColor");
    locA_I = glGetUniformLocation(progInst, "uAlpha");
    locVP_G = glGetUniformLocation(progGround, "uViewProj");
    locM_G = glGetUniformLocation(progGround, "uModel");
    locGrassTile_G = glGetUniformLocation(progGround, "uGrassTileM");
    locNoiseTile_G = glGetUniformLocation(progGround, "uNoiseTileM");
    locGrassTex_G = glGetUniformLocation(progGround, "uGrassTex");
    locNoiseTex_G = glGetUniformLocation(progGround, "uNoiseTex");
    locVP_S = glGetUniformLocation(progSky, "uViewProj");
    locSkyTex_S = glGetUniformLocation(progSky, "uSkybox");
    if (locVP_B < 0 || locM_B < 0 || locC_B < 0 || locA_B < 0 ||
        locVP_I < 0 || locC_I < 0 || locA_I < 0 ||
        locVP_G < 0 || locM_G < 0 || locGrassTile_G < 0 || locNoiseTile_G < 0 ||
        locGrassTex_G < 0 || locNoiseTex_G < 0 ||
        locVP_S < 0 || locSkyTex_S < 0) {
        SDL_Log("Renderer init failed: missing uniforms.");
        return false;
    }

    bool grassOk = false;
    bool noiseOk = false;
    texGrass = LoadTexture2D("assets/textures/grass.png", 80, 110, 70, 255, &grassOk);
    texNoise = LoadTexture2D("assets/textures/grayscale.png", 128, 128, 128, 255, &noiseOk);
    if (!grassOk || !noiseOk) {
        SDL_Log("Renderer: using fallback ground texture(s).");
    }

    const char* skyFaces[6] = {
        "assets/textures/Daylight Box_Right.png",
        "assets/textures/Daylight Box_Left.png",
        "assets/textures/Daylight Box_Top.png",
        "assets/textures/Daylight Box_Bottom.png",
        "assets/textures/Daylight Box_Front.png",
        "assets/textures/Daylight Box_Back.png"
    };
    bool skyOk = false;
    texSkybox = LoadCubemap(skyFaces, &skyOk);
    if (!skyOk) {
        SDL_Log("Renderer: using fallback skybox texture.");
    }

    // Ground quad
    const float MAP_SIDE_M = 105500.0f;
    const float HALF = MAP_SIDE_M * 0.5f;
    glm::vec3 groundVerts[6] = {
        {-HALF, 0.0f, -HALF},
        { HALF, 0.0f, -HALF},
        { HALF, 0.0f,  HALF},
        {-HALF, 0.0f, -HALF},
        { HALF, 0.0f,  HALF},
        {-HALF, 0.0f,  HALF},
    };

    glGenVertexArrays(1, &vaoGround);
    glGenBuffers(1, &vboGround);
    glBindVertexArray(vaoGround);
    glBindBuffer(GL_ARRAY_BUFFER, vboGround);
    glBufferData(GL_ARRAY_BUFFER, sizeof(groundVerts), groundVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    // Dynamic buffers
    glGenVertexArrays(1, &vaoRoad);
    glGenBuffers(1, &vboRoad);
    glBindVertexArray(vaoRoad);
    glBindBuffer(GL_ARRAY_BUFFER, vboRoad);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoPreview);
    glGenBuffers(1, &vboPreview);
    glBindVertexArray(vaoPreview);
    glBindBuffer(GL_ARRAY_BUFFER, vboPreview);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    // Cube mesh
    const glm::vec3 cube[36] = {
        {-0.5f,-0.5f, 0.5f},{ 0.5f,-0.5f, 0.5f},{ 0.5f, 0.5f, 0.5f},
        {-0.5f,-0.5f, 0.5f},{ 0.5f, 0.5f, 0.5f},{-0.5f, 0.5f, 0.5f},
        { 0.5f,-0.5f,-0.5f},{-0.5f,-0.5f,-0.5f},{-0.5f, 0.5f,-0.5f},
        { 0.5f,-0.5f,-0.5f},{-0.5f, 0.5f,-0.5f},{ 0.5f, 0.5f,-0.5f},
        { 0.5f,-0.5f, 0.5f},{ 0.5f,-0.5f,-0.5f},{ 0.5f, 0.5f,-0.5f},
        { 0.5f,-0.5f, 0.5f},{ 0.5f, 0.5f,-0.5f},{ 0.5f, 0.5f, 0.5f},
        {-0.5f,-0.5f,-0.5f},{-0.5f,-0.5f, 0.5f},{-0.5f, 0.5f, 0.5f},
        {-0.5f,-0.5f,-0.5f},{-0.5f, 0.5f, 0.5f},{-0.5f, 0.5f,-0.5f},
        {-0.5f, 0.5f, 0.5f},{ 0.5f, 0.5f, 0.5f},{ 0.5f, 0.5f,-0.5f},
        {-0.5f, 0.5f, 0.5f},{ 0.5f, 0.5f,-0.5f},{-0.5f, 0.5f,-0.5f},
        {-0.5f,-0.5f,-0.5f},{ 0.5f,-0.5f,-0.5f},{ 0.5f,-0.5f, 0.5f},
        {-0.5f,-0.5f,-0.5f},{ 0.5f,-0.5f, 0.5f},{-0.5f,-0.5f, 0.5f},
    };

    glGenBuffers(1, &vboCube);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube), cube, GL_STATIC_DRAW);

    glGenVertexArrays(1, &vaoSkybox);
    glBindVertexArray(vaoSkybox);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoCubeSingle);
    glBindVertexArray(vaoCubeSingle);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoCubeInstAnim);
    glGenBuffers(1, &vboInstAnim);

    glBindVertexArray(vaoCubeInstAnim);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vboInstAnim);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    SetupInstanceAttribs(vaoCubeInstAnim, vboInstAnim);

    glBindVertexArray(0);
    return true;
}

void Renderer::resize(int w, int h) {
    glViewport(0, 0, w, h);
}

void Renderer::updateRoadMesh(const std::vector<glm::vec3>& verts) {
    UploadDynamicVerts(vboRoad, capRoad, verts);
}

void Renderer::updatePreviewMesh(const std::vector<glm::vec3>& verts) {
    UploadDynamicVerts(vboPreview, capPreview, verts);
}

void Renderer::updateHouseChunk(uint64_t key, AssetId assetId, const MeshGpu& mesh, const std::vector<HouseInstanceGPU>& instances) {
    auto& assetMap = houseChunks[key];
    auto& buf = assetMap[assetId];
    if (buf.vao == 0 || buf.meshVbo != mesh.vbo || buf.meshEbo != mesh.ebo) {
        if (buf.vao == 0) glGenVertexArrays(1, &buf.vao);
        if (buf.vbo == 0) glGenBuffers(1, &buf.vbo);
        glBindVertexArray(buf.vao);
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glEnableVertexAttribArray(0);
        GLsizei stride = mesh.vertexStride > 0 ? mesh.vertexStride : (GLsizei)sizeof(glm::vec3);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        if (mesh.indexed && mesh.ebo) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        } else {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        glBindBuffer(GL_ARRAY_BUFFER, buf.vbo);
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
        SetupInstanceAttribs(buf.vao, buf.vbo);
        glBindVertexArray(0);

        buf.meshVbo = mesh.vbo;
        buf.meshEbo = mesh.ebo;
        buf.vertexCount = mesh.vertexCount;
        buf.indexCount = mesh.indexCount;
        buf.indexed = mesh.indexed;
    } else {
        buf.vertexCount = mesh.vertexCount;
        buf.indexCount = mesh.indexCount;
        buf.indexed = mesh.indexed;
    }

    glBindBuffer(GL_ARRAY_BUFFER, buf.vbo);
    std::size_t bytes = instances.size() * sizeof(HouseInstanceGPU);
    if (bytes == 0) {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
        buf.capacity = 0;
        buf.count = 0;
    } else {
        if (bytes > buf.capacity) {
            buf.capacity = (std::size_t)(bytes * 1.5f) + 256;
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)buf.capacity, nullptr, GL_DYNAMIC_DRAW);
        } else {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)buf.capacity, nullptr, GL_DYNAMIC_DRAW); // orphan
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)bytes, instances.data());
        buf.count = instances.size();
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::updateAnimHouses(const std::vector<HouseInstanceGPU>& animHouses) {
    glBindBuffer(GL_ARRAY_BUFFER, vboInstAnim);
    std::size_t bytes = animHouses.size() * sizeof(HouseInstanceGPU);
    if (bytes == 0) {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
        capInstAnim = 0;
    } else {
        if (bytes > capInstAnim) {
            capInstAnim = (std::size_t)(bytes * 1.5f) + 256;
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)capInstAnim, nullptr, GL_DYNAMIC_DRAW);
        } else {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)capInstAnim, nullptr, GL_DYNAMIC_DRAW); // orphan
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)bytes, animHouses.data());
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::render(const RenderFrame& frame) {
    glClearColor(0.55f, 0.75f, 0.95f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Ground/roads/overlays are single-sided quads; render without culling
    glDisable(GL_CULL_FACE);

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(progSky);
    glUniformMatrix4fv(locVP_S, 1, GL_FALSE, &frame.viewProjSky[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texSkybox);
    glUniform1i(locSkyTex_S, 0);
    glBindVertexArray(vaoSkybox);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    glm::mat4 I(1.0f);
    glUseProgram(progGround);
    glUniformMatrix4fv(locVP_G, 1, GL_FALSE, &frame.viewProj[0][0]);
    glUniformMatrix4fv(locM_G, 1, GL_FALSE, &I[0][0]);
    glUniform1f(locGrassTile_G, 4.0f);
    glUniform1f(locNoiseTile_G, 96.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texGrass);
    glUniform1i(locGrassTex_G, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texNoise);
    glUniform1i(locNoiseTex_G, 1);
    glBindVertexArray(vaoGround);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(progBasic);
    glUniformMatrix4fv(locVP_B, 1, GL_FALSE, &frame.viewProj[0][0]);
    glUniformMatrix4fv(locM_B, 1, GL_FALSE, &I[0][0]);
    glUniform3f(locC_B, 0.18f, 0.18f, 0.18f);
    glUniform1f(locA_B, 1.0f);
    glBindVertexArray(vaoRoad);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)frame.roadVertexCount);

    // Preview overlay
    std::size_t overlayTotal = frame.zoneResidentialVertexCount
        + frame.zoneCommercialVertexCount
        + frame.zoneIndustrialVertexCount
        + frame.zoneOfficeVertexCount;
    if (frame.gridVertexCount + overlayTotal + frame.previewVertexCount > 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);

        glBindVertexArray(vaoPreview);

        // Lot grid
        if (frame.gridVertexCount > 0) {
            glUniform3f(locC_B, 0.10f, 0.60f, 0.75f);
            glUniform1f(locA_B, 0.15f);
            glDrawArrays(GL_TRIANGLES, 0, (GLsizei)frame.gridVertexCount);
        }

        auto applyZoneColor = [&](uint8_t zoneType) {
            switch (zoneType) {
                case 1: glUniform3f(locC_B, 0.20f, 0.45f, 0.90f); break; // commercial (blue)
                case 2: glUniform3f(locC_B, 0.85f, 0.75f, 0.20f); break; // industrial (yellow)
                case 3: glUniform3f(locC_B, 0.45f, 0.75f, 0.95f); break; // office (light blue)
                default: glUniform3f(locC_B, 0.15f, 0.65f, 0.35f); break; // residential
            }
        };

        // Existing zones overlay (by type)
        GLint offset = (GLint)frame.gridVertexCount;
        if (frame.zoneResidentialVertexCount > 0) {
            applyZoneColor(0);
            glUniform1f(locA_B, 0.30f);
            glDrawArrays(GL_TRIANGLES, offset, (GLsizei)frame.zoneResidentialVertexCount);
            offset += (GLint)frame.zoneResidentialVertexCount;
        }
        if (frame.zoneCommercialVertexCount > 0) {
            applyZoneColor(1);
            glUniform1f(locA_B, 0.30f);
            glDrawArrays(GL_TRIANGLES, offset, (GLsizei)frame.zoneCommercialVertexCount);
            offset += (GLint)frame.zoneCommercialVertexCount;
        }
        if (frame.zoneIndustrialVertexCount > 0) {
            applyZoneColor(2);
            glUniform1f(locA_B, 0.30f);
            glDrawArrays(GL_TRIANGLES, offset, (GLsizei)frame.zoneIndustrialVertexCount);
            offset += (GLint)frame.zoneIndustrialVertexCount;
        }
        if (frame.zoneOfficeVertexCount > 0) {
            applyZoneColor(3);
            glUniform1f(locA_B, 0.30f);
            glDrawArrays(GL_TRIANGLES, offset, (GLsizei)frame.zoneOfficeVertexCount);
            offset += (GLint)frame.zoneOfficeVertexCount;
        }

        // Active preview
        if (frame.previewVertexCount > 0) {
            if (frame.drawRoadPreview) {
                glUniform3f(locC_B, 0.20f, 0.65f, 0.95f);
                glUniform1f(locA_B, 0.50f);
            } else {
                if (frame.zonePreviewValid) applyZoneColor(frame.zonePreviewType);
                else glUniform3f(locC_B, 0.90f, 0.20f, 0.20f);
                glUniform1f(locA_B, 0.35f);
            }
            glDrawArrays(GL_TRIANGLES, offset, (GLsizei)frame.previewVertexCount);
        }

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    // Markers
    for (const auto& m : frame.markers) {
        glm::mat4 M(1.0f);
        M = glm::translate(M, glm::vec3(m.pos.x, m.pos.y + 0.4f, m.pos.z));
        M = glm::scale(M, glm::vec3(m.scale, m.scale, m.scale));
        glUniformMatrix4fv(locM_B, 1, GL_FALSE, &M[0][0]);
        glUniform3f(locC_B, m.color.x, m.color.y, m.color.z);
        glUniform1f(locA_B, 1.0f);
        glBindVertexArray(vaoCubeSingle);
        glDrawArrays(GL_TRIANGLES, 0, 36);
    }

    // Houses
    glUseProgram(progInst);
    glUniformMatrix4fv(locVP_I, 1, GL_FALSE, &frame.viewProj[0][0]);
    glUniform3f(locC_I, 0.75f, 0.72f, 0.62f);
    glUniform1f(locA_I, 1.0f);

    // Houses are closed meshes; enable culling here for perf
    glEnable(GL_CULL_FACE);

    for (const auto& batch : frame.visibleHouseBatches) {
        auto chunkIt = houseChunks.find(batch.chunkKey);
        if (chunkIt == houseChunks.end()) continue;
        auto assetIt = chunkIt->second.find(batch.asset);
        if (assetIt == chunkIt->second.end()) continue;
        const ChunkBuf& buf = assetIt->second;
        if (buf.count == 0) continue;
        glBindVertexArray(buf.vao);
        if (buf.indexed) {
            glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)buf.indexCount, GL_UNSIGNED_INT, (void*)0, (GLsizei)buf.count);
        } else {
            glDrawArraysInstanced(GL_TRIANGLES, 0, (GLsizei)buf.vertexCount, (GLsizei)buf.count);
        }
    }

    if (frame.houseAnimCount > 0) {
        glBindVertexArray(vaoCubeInstAnim);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, (GLsizei)frame.houseAnimCount);
    }

    glBindVertexArray(0);
}

void Renderer::destroyGL() {
    if (progBasic) { glDeleteProgram(progBasic); progBasic = 0; }
    if (progInst) { glDeleteProgram(progInst); progInst = 0; }
    if (progGround) { glDeleteProgram(progGround); progGround = 0; }
    if (progSky) { glDeleteProgram(progSky); progSky = 0; }
    if (texGrass) { glDeleteTextures(1, &texGrass); texGrass = 0; }
    if (texNoise) { glDeleteTextures(1, &texNoise); texNoise = 0; }
    if (texSkybox) { glDeleteTextures(1, &texSkybox); texSkybox = 0; }

    GLuint vaos[] = { vaoGround, vaoRoad, vaoPreview, vaoSkybox, vaoCubeSingle, vaoCubeInstAnim };
    GLuint vbos[] = { vboGround, vboRoad, vboPreview, vboCube, vboInstAnim };

    glDeleteVertexArrays((GLsizei)std::size(vaos), vaos);
    glDeleteBuffers((GLsizei)std::size(vbos), vbos);

    for (auto& chunkKv : houseChunks) {
        for (auto& assetKv : chunkKv.second) {
            if (assetKv.second.vao) glDeleteVertexArrays(1, &assetKv.second.vao);
            if (assetKv.second.vbo) glDeleteBuffers(1, &assetKv.second.vbo);
        }
    }
    houseChunks.clear();

    vaoGround = vaoRoad = vaoPreview = vaoSkybox = vaoCubeSingle = vaoCubeInstAnim = 0;
    vboGround = vboRoad = vboPreview = vboCube = vboInstAnim = 0;
}

void Renderer::shutdown() {
    destroyGL();
}
