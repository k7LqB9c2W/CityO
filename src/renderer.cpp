#include "renderer.h"

#include "config.h"
#include "image_loader.h"
#include "mesh_cache.h"

#include <SDL.h>
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct VertexPN {
    glm::vec3 pos;
    glm::vec3 normal;
};

GLuint MakeProgram(const char* vsSrc, const char* fsSrc);
bool GLCheckShader(GLuint shader, const char* label);
bool GLCheckProgram(GLuint prog);
void SetupInstanceAttribs(GLuint vao, GLuint instanceVbo);
void UploadDynamicVerts(GLuint vbo, std::size_t& capacityBytes, const std::vector<glm::vec3>& verts);
void UploadDynamicRoadVerts(GLuint vbo, std::size_t& capacityBytes, const std::vector<RoadVertex>& verts);
void UploadDynamicMats(GLuint vbo, std::size_t& capacityBytes, const std::vector<glm::mat4>& mats);
GLuint CreateTextureFromRGBA(const uint8_t* pixels, int w, int h, bool srgb);
GLuint CreateSolidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb);
GLuint LoadTexture2D(const char* path, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb, bool* outOk);
GLuint CreateSolidCubemap(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb);
GLuint LoadCubemap(const char* faces[6], bool srgb, bool* outOk);
bool CreateShadowMap(int size, GLuint& outFbo, GLuint& outTex);

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

void UploadDynamicRoadVerts(GLuint vbo, std::size_t& capacityBytes, const std::vector<RoadVertex>& verts) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    std::size_t bytes = verts.size() * sizeof(RoadVertex);
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

GLuint CreateTextureFromRGBA(const uint8_t* pixels, int w, int h, bool srgb) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GLenum internalFormat = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
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

GLuint CreateSolidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb) {
    uint8_t pixel[4] = { r, g, b, a };
    return CreateTextureFromRGBA(pixel, 1, 1, srgb);
}

GLuint LoadTexture2D(const char* path, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb, bool* outOk) {
    std::vector<uint8_t> pixels;
    int w = 0;
    int h = 0;
    if (LoadImageRGBA(path, pixels, w, h)) {
        if (outOk) *outOk = true;
        return CreateTextureFromRGBA(pixels.data(), w, h, srgb);
    }
    if (outOk) *outOk = false;
    SDL_Log("Texture load failed: %s", path);
    return CreateSolidTexture(r, g, b, a, srgb);
}

GLuint CreateSolidCubemap(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    uint8_t pixel[4] = { r, g, b, a };
    GLenum internalFormat = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internalFormat, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return tex;
}

GLuint LoadCubemap(const char* faces[6], bool srgb, bool* outOk) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    bool ok = true;
    int w = 0;
    int h = 0;
    std::vector<uint8_t> pixels;
    GLenum internalFormat = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    for (int i = 0; i < 6; ++i) {
        int wi = 0;
        int hi = 0;
        if (!LoadImageRGBA(faces[i], pixels, wi, hi)) {
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
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internalFormat, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    if (!ok) {
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glDeleteTextures(1, &tex);
        if (outOk) *outOk = false;
        return CreateSolidCubemap(120, 160, 210, 255, srgb);
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

bool CreateShadowMap(int size, GLuint& outFbo, GLuint& outTex) {
    glGenTextures(1, &outTex);
    glBindTexture(GL_TEXTURE_2D, outTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size, size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &outFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, outFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, outTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (!ok) {
        glDeleteTextures(1, &outTex);
        glDeleteFramebuffers(1, &outFbo);
        outTex = 0;
        outFbo = 0;
    }
    return ok;
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
        layout(location=1) in vec3 aNormal;
        layout(location=2) in vec4 iPosYaw;   // xyz, yaw
        layout(location=3) in vec4 iScaleVar; // xyz scale
        uniform mat4 uViewProj;
        uniform mat4 uLightViewProj;
        out vec3 vNormal;
        out vec4 vLightPos;
        out vec3 vLocalPos;
        out vec3 vLocalNormal;
        flat out float vFacadeIndex;
        flat out vec3 vScale;
        void main() {
            float yaw = iPosYaw.w;
            mat3 R = mat3(
                cos(yaw), 0.0, -sin(yaw),
                0.0,      1.0,  0.0,
                sin(yaw), 0.0,  cos(yaw)
            );
            vec3 scale = max(iScaleVar.xyz, vec3(0.0001));
            vec3 localPos = aPos * scale;
            vec3 scaled = R * localPos;
            vec3 worldPos = iPosYaw.xyz + scaled;
            worldPos.y += 0.05; // lift houses off the ground to avoid z-fighting
            gl_Position = uViewProj * vec4(worldPos, 1.0);
            vec3 invScale = 1.0 / scale;
            vNormal = normalize(R * (aNormal * invScale));
            vLightPos = uLightViewProj * vec4(worldPos, 1.0);
            vLocalPos = localPos;
            vLocalNormal = aNormal;
            vFacadeIndex = iScaleVar.w;
            vScale = scale;
        }
    )";

    const char* vsGround = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uViewProj;
        uniform mat4 uModel;
        uniform float uGrassTileM;
        uniform float uNoiseTileM;
        uniform mat4 uLightViewProj;
        out vec2 vGrassUV;
        out vec2 vNoiseUV;
        out vec3 vNormal;
        out vec4 vLightPos;
        void main() {
            vec4 world = uModel * vec4(aPos, 1.0);
            vGrassUV = world.xz / uGrassTileM;
            vNoiseUV = world.xz / uNoiseTileM;
            vNormal = vec3(0.0, 1.0, 0.0);
            vLightPos = uLightViewProj * world;
            gl_Position = uViewProj * world;
        }
    )";

    const char* vsRoad = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec2 aUV;
        uniform mat4 uViewProj;
        uniform mat4 uLightViewProj;
        out vec2 vUV;
        out vec3 vNormal;
        out vec4 vLightPos;
        void main() {
            vec4 world = vec4(aPos, 1.0);
            vUV = aUV;
            vNormal = vec3(0.0, 1.0, 0.0);
            vLightPos = uLightViewProj * world;
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
        uniform float uExposure;
        vec3 ToneMap(vec3 color) {
            color *= uExposure;
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            return color;
        }
        void main() {
            FragColor = vec4(ToneMap(uColor), uAlpha);
        }
    )";

    const char* fsSky = R"(
        #version 330 core
        in vec3 vDir;
        out vec4 FragColor;
        uniform samplerCube uSkybox;
        uniform float uSkyBrightness;
        uniform float uExposure;
        uniform float uSkyExposure;
        vec3 ToneMap(vec3 color) {
            color *= (uExposure * uSkyExposure);
            color = pow(color, vec3(1.0 / 2.2));
            return color;
        }
        void main() {
            vec3 color = texture(uSkybox, normalize(vDir)).rgb * uSkyBrightness;
            FragColor = vec4(ToneMap(color), 1.0);
        }
    )";

    const char* fsGround = R"(
        #version 330 core
        in vec2 vGrassUV;
        in vec2 vNoiseUV;
        in vec3 vNormal;
        in vec4 vLightPos;
        out vec4 FragColor;
        uniform sampler2D uGrassTex;
        uniform sampler2D uNoiseTex;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform float uSunIntensity;
        uniform vec3 uAmbientColor;
        uniform float uAmbientIntensity;
        uniform float uExposure;
        uniform sampler2DShadow uShadowMap;
        uniform vec2 uShadowTexel;
        uniform float uShadowStrength;
        vec3 ToneMap(vec3 color) {
            color *= uExposure;
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            return color;
        }
        float ShadowVisibility(vec4 lightPos, vec3 normal) {
            if (uShadowStrength <= 0.0) return 1.0;
            vec3 proj = lightPos.xyz / lightPos.w;
            proj = proj * 0.5 + 0.5;
            if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
                return 1.0;
            }
            float ndotl = max(dot(normal, uSunDir), 0.0);
            float bias = max(0.0015 * (1.0 - ndotl), 0.0005);
            float shadow = 0.0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    vec2 offset = vec2(x, y) * uShadowTexel;
                    shadow += texture(uShadowMap, vec3(proj.xy + offset, proj.z - bias));
                }
            }
            shadow /= 9.0;
            return mix(1.0, shadow, uShadowStrength);
        }
        void main() {
            vec3 grass = texture(uGrassTex, vGrassUV).rgb;
            float n = texture(uNoiseTex, vNoiseUV).r;
            float shade = mix(0.85, 1.15, n);
            vec3 base = grass * shade;
            vec3 normal = normalize(vNormal);
            float ndotl = max(dot(normal, uSunDir), 0.0);
            float shadow = ShadowVisibility(vLightPos, normal);
            vec3 ambient = uAmbientColor * uAmbientIntensity;
            vec3 direct = uSunColor * uSunIntensity * ndotl * shadow;
            vec3 color = base * (ambient + direct);
            FragColor = vec4(ToneMap(color), 1.0);
        }
    )";

    const char* fsRoad = R"(
        #version 330 core
        in vec2 vUV;
        in vec3 vNormal;
        in vec4 vLightPos;
        out vec4 FragColor;
        uniform sampler2D uRoadTex;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform float uSunIntensity;
        uniform vec3 uAmbientColor;
        uniform float uAmbientIntensity;
        uniform float uExposure;
        uniform sampler2DShadow uShadowMap;
        uniform vec2 uShadowTexel;
        uniform float uShadowStrength;
        vec3 ToneMap(vec3 color) {
            color *= uExposure;
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            return color;
        }
        float ShadowVisibility(vec4 lightPos, vec3 normal) {
            if (uShadowStrength <= 0.0) return 1.0;
            vec3 proj = lightPos.xyz / lightPos.w;
            proj = proj * 0.5 + 0.5;
            if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
                return 1.0;
            }
            float ndotl = max(dot(normal, uSunDir), 0.0);
            float bias = max(0.0015 * (1.0 - ndotl), 0.0005);
            float shadow = 0.0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    vec2 offset = vec2(x, y) * uShadowTexel;
                    shadow += texture(uShadowMap, vec3(proj.xy + offset, proj.z - bias));
                }
            }
            shadow /= 9.0;
            return mix(1.0, shadow, uShadowStrength);
        }
        void main() {
            vec3 base = texture(uRoadTex, vUV).rgb;
            vec3 normal = normalize(vNormal);
            float ndotl = max(dot(normal, uSunDir), 0.0);
            float shadow = ShadowVisibility(vLightPos, normal);
            vec3 ambient = uAmbientColor * uAmbientIntensity;
            vec3 direct = uSunColor * uSunIntensity * ndotl * shadow;
            vec3 color = base * (ambient + direct);
            FragColor = vec4(ToneMap(color), 1.0);
        }
    )";

    const char* fsInst = R"(
        #version 330 core
        in vec3 vNormal;
        in vec4 vLightPos;
        in vec3 vLocalPos;
        in vec3 vLocalNormal;
        flat in float vFacadeIndex;
        flat in vec3 vScale;
        out vec4 FragColor;
        uniform vec3 uColor;
        uniform float uAlpha;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform float uSunIntensity;
        uniform vec3 uAmbientColor;
        uniform float uAmbientIntensity;
        uniform float uExposure;
        uniform sampler2DShadow uShadowMap;
        uniform vec2 uShadowTexel;
        uniform float uShadowStrength;
        uniform sampler2D uFacadeTex0;
        uniform sampler2D uFacadeTex1;
        uniform sampler2D uFacadeTex2;
        uniform sampler2D uFacadeTex3;
        uniform vec2 uFacadeTileM;
        uniform vec3 uFacadeTint;
        vec3 ToneMap(vec3 color) {
            color *= uExposure;
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            return color;
        }
        float ShadowVisibility(vec4 lightPos, vec3 normal) {
            if (uShadowStrength <= 0.0) return 1.0;
            vec3 proj = lightPos.xyz / lightPos.w;
            proj = proj * 0.5 + 0.5;
            if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
                return 1.0;
            }
            float ndotl = max(dot(normal, uSunDir), 0.0);
            float bias = max(0.0015 * (1.0 - ndotl), 0.0005);
            float shadow = 0.0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    vec2 offset = vec2(x, y) * uShadowTexel;
                    shadow += texture(uShadowMap, vec3(proj.xy + offset, proj.z - bias));
                }
            }
            shadow /= 9.0;
            return mix(1.0, shadow, uShadowStrength);
        }
        void main() {
            vec3 normal = normalize(vNormal);
            float ndotl = max(dot(normal, uSunDir), 0.0);
            float shadow = ShadowVisibility(vLightPos, normal);
            vec3 ambient = uAmbientColor * uAmbientIntensity;
            vec3 direct = uSunColor * uSunIntensity * ndotl * shadow;
            vec3 baseColor = uColor;
            if (vFacadeIndex >= 0.0) {
                vec3 ln = normalize(vLocalNormal);
                if (abs(ln.y) < 0.9) {
                    float v = (vLocalPos.y + vScale.y * 0.5) / uFacadeTileM.y;
                    float u = 0.0;
                    if (abs(ln.x) > abs(ln.z)) {
                        float halfWidth = vScale.z * 0.5;
                        float horiz = vLocalPos.z;
                        u = (ln.x > 0.0) ? (horiz + halfWidth) : (halfWidth - horiz);
                    } else {
                        float halfWidth = vScale.x * 0.5;
                        float horiz = vLocalPos.x;
                        u = (ln.z > 0.0) ? (horiz + halfWidth) : (halfWidth - horiz);
                    }
                    vec2 uv = vec2(u / uFacadeTileM.x, v);
                    int idx = int(clamp(vFacadeIndex, 0.0, 3.0) + 0.5);
                    vec3 facade = texture(uFacadeTex0, uv).rgb;
                    if (idx == 1) facade = texture(uFacadeTex1, uv).rgb;
                    else if (idx == 2) facade = texture(uFacadeTex2, uv).rgb;
                    else if (idx == 3) facade = texture(uFacadeTex3, uv).rgb;
                    baseColor = facade * uFacadeTint;
                }
            }
            vec3 color = baseColor * (ambient + direct);
            FragColor = vec4(ToneMap(color), uAlpha);
        }
    )";

    const char* vsDepth = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uLightViewProj;
        uniform mat4 uModel;
        void main() {
            gl_Position = uLightViewProj * uModel * vec4(aPos, 1.0);
        }
    )";

    const char* vsDepthInst = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=2) in vec4 iPosYaw;
        layout(location=3) in vec4 iScaleVar;
        uniform mat4 uLightViewProj;
        void main() {
            float yaw = iPosYaw.w;
            mat3 R = mat3(
                cos(yaw), 0.0, -sin(yaw),
                0.0,      1.0,  0.0,
                sin(yaw), 0.0,  cos(yaw)
            );
            vec3 scale = max(iScaleVar.xyz, vec3(0.0001));
            vec3 scaled = R * (aPos * scale);
            vec3 worldPos = iPosYaw.xyz + scaled;
            worldPos.y += 0.05;
            gl_Position = uLightViewProj * vec4(worldPos, 1.0);
        }
    )";

    const char* fsDepth = R"(
        #version 330 core
        void main() {
        }
    )";

    progBasic = MakeProgram(vsBasic, fsColor);
    progInst = MakeProgram(vsInstanced, fsInst);
    progGround = MakeProgram(vsGround, fsGround);
    progRoad = MakeProgram(vsRoad, fsRoad);
    progSky = MakeProgram(vsSky, fsSky);
    progDepth = MakeProgram(vsDepth, fsDepth);
    progDepthInst = MakeProgram(vsDepthInst, fsDepth);
    if (!progBasic || !progInst || !progGround || !progRoad || !progSky || !progDepth || !progDepthInst) return false;

    locVP_B = glGetUniformLocation(progBasic, "uViewProj");
    locM_B = glGetUniformLocation(progBasic, "uModel");
    locC_B = glGetUniformLocation(progBasic, "uColor");
    locA_B = glGetUniformLocation(progBasic, "uAlpha");
    locExposure_B = glGetUniformLocation(progBasic, "uExposure");

    locVP_I = glGetUniformLocation(progInst, "uViewProj");
    locC_I = glGetUniformLocation(progInst, "uColor");
    locA_I = glGetUniformLocation(progInst, "uAlpha");
    locSunDir_I = glGetUniformLocation(progInst, "uSunDir");
    locSunColor_I = glGetUniformLocation(progInst, "uSunColor");
    locSunInt_I = glGetUniformLocation(progInst, "uSunIntensity");
    locAmbColor_I = glGetUniformLocation(progInst, "uAmbientColor");
    locAmbInt_I = glGetUniformLocation(progInst, "uAmbientIntensity");
    locExposure_I = glGetUniformLocation(progInst, "uExposure");
    locLightVP_I = glGetUniformLocation(progInst, "uLightViewProj");
    locShadowMap_I = glGetUniformLocation(progInst, "uShadowMap");
    locShadowTexel_I = glGetUniformLocation(progInst, "uShadowTexel");
    locShadowStrength_I = glGetUniformLocation(progInst, "uShadowStrength");
    locFacadeTex0_I = glGetUniformLocation(progInst, "uFacadeTex0");
    locFacadeTex1_I = glGetUniformLocation(progInst, "uFacadeTex1");
    locFacadeTex2_I = glGetUniformLocation(progInst, "uFacadeTex2");
    locFacadeTex3_I = glGetUniformLocation(progInst, "uFacadeTex3");
    locFacadeTile_I = glGetUniformLocation(progInst, "uFacadeTileM");
    locFacadeTint_I = glGetUniformLocation(progInst, "uFacadeTint");
    locVP_G = glGetUniformLocation(progGround, "uViewProj");
    locM_G = glGetUniformLocation(progGround, "uModel");
    locGrassTile_G = glGetUniformLocation(progGround, "uGrassTileM");
    locNoiseTile_G = glGetUniformLocation(progGround, "uNoiseTileM");
    locGrassTex_G = glGetUniformLocation(progGround, "uGrassTex");
    locNoiseTex_G = glGetUniformLocation(progGround, "uNoiseTex");
    locSunDir_G = glGetUniformLocation(progGround, "uSunDir");
    locSunColor_G = glGetUniformLocation(progGround, "uSunColor");
    locSunInt_G = glGetUniformLocation(progGround, "uSunIntensity");
    locAmbColor_G = glGetUniformLocation(progGround, "uAmbientColor");
    locAmbInt_G = glGetUniformLocation(progGround, "uAmbientIntensity");
    locExposure_G = glGetUniformLocation(progGround, "uExposure");
    locLightVP_G = glGetUniformLocation(progGround, "uLightViewProj");
    locShadowMap_G = glGetUniformLocation(progGround, "uShadowMap");
    locShadowTexel_G = glGetUniformLocation(progGround, "uShadowTexel");
    locShadowStrength_G = glGetUniformLocation(progGround, "uShadowStrength");
    locVP_R = glGetUniformLocation(progRoad, "uViewProj");
    locLightVP_R = glGetUniformLocation(progRoad, "uLightViewProj");
    locRoadTex_R = glGetUniformLocation(progRoad, "uRoadTex");
    locSunDir_R = glGetUniformLocation(progRoad, "uSunDir");
    locSunColor_R = glGetUniformLocation(progRoad, "uSunColor");
    locSunInt_R = glGetUniformLocation(progRoad, "uSunIntensity");
    locAmbColor_R = glGetUniformLocation(progRoad, "uAmbientColor");
    locAmbInt_R = glGetUniformLocation(progRoad, "uAmbientIntensity");
    locExposure_R = glGetUniformLocation(progRoad, "uExposure");
    locShadowMap_R = glGetUniformLocation(progRoad, "uShadowMap");
    locShadowTexel_R = glGetUniformLocation(progRoad, "uShadowTexel");
    locShadowStrength_R = glGetUniformLocation(progRoad, "uShadowStrength");
    locVP_S = glGetUniformLocation(progSky, "uViewProj");
    locSkyTex_S = glGetUniformLocation(progSky, "uSkybox");
    locSkyBright_S = glGetUniformLocation(progSky, "uSkyBrightness");
    locExposure_S = glGetUniformLocation(progSky, "uExposure");
    locSkyExposure_S = glGetUniformLocation(progSky, "uSkyExposure");
    locLightVP_D = glGetUniformLocation(progDepth, "uLightViewProj");
    locM_D = glGetUniformLocation(progDepth, "uModel");
    locLightVP_DI = glGetUniformLocation(progDepthInst, "uLightViewProj");
    if (locVP_B < 0 || locM_B < 0 || locC_B < 0 || locA_B < 0 || locExposure_B < 0 ||
        locVP_I < 0 || locC_I < 0 || locA_I < 0 || locSunDir_I < 0 || locSunColor_I < 0 ||
        locSunInt_I < 0 || locAmbColor_I < 0 || locAmbInt_I < 0 || locExposure_I < 0 ||
        locLightVP_I < 0 || locShadowMap_I < 0 || locShadowTexel_I < 0 || locShadowStrength_I < 0 ||
        locFacadeTex0_I < 0 || locFacadeTex1_I < 0 || locFacadeTex2_I < 0 || locFacadeTex3_I < 0 ||
        locFacadeTile_I < 0 || locFacadeTint_I < 0 ||
        locVP_G < 0 || locM_G < 0 || locGrassTile_G < 0 || locNoiseTile_G < 0 ||
        locGrassTex_G < 0 || locNoiseTex_G < 0 || locSunDir_G < 0 || locSunColor_G < 0 ||
        locSunInt_G < 0 || locAmbColor_G < 0 || locAmbInt_G < 0 || locExposure_G < 0 ||
        locLightVP_G < 0 || locShadowMap_G < 0 || locShadowTexel_G < 0 || locShadowStrength_G < 0 ||
        locVP_R < 0 || locLightVP_R < 0 || locRoadTex_R < 0 || locSunDir_R < 0 ||
        locSunColor_R < 0 || locSunInt_R < 0 || locAmbColor_R < 0 || locAmbInt_R < 0 ||
        locExposure_R < 0 || locShadowMap_R < 0 || locShadowTexel_R < 0 || locShadowStrength_R < 0 ||
        locVP_S < 0 || locSkyTex_S < 0 || locSkyBright_S < 0 || locExposure_S < 0 ||
        locSkyExposure_S < 0 ||
        locLightVP_D < 0 || locM_D < 0 || locLightVP_DI < 0) {
        SDL_Log("Renderer init failed: missing uniforms.");
        return false;
    }

    if (!CreateShadowMap(shadowMapSize, shadowFbo, shadowTex)) {
        SDL_Log("Renderer: shadow map init failed, shadows disabled.");
    }

    bool grassOk = false;
    bool noiseOk = false;
    texGrass = LoadTexture2D("assets/textures/grass.png", 80, 110, 70, 255, true, &grassOk);
    texNoise = LoadTexture2D("assets/textures/grayscale.png", 128, 128, 128, 255, false, &noiseOk);
    if (!grassOk || !noiseOk) {
        SDL_Log("Renderer: using fallback ground texture(s).");
    }
    bool waterOk = false;
    texWater = LoadTexture2D("assets/textures/water.png", 40, 80, 120, 255, true, &waterOk);
    if (!waterOk) {
        SDL_Log("Renderer: using fallback water texture.");
    }
    bool roadOk = false;
    texRoad = LoadTexture2D("assets/textures/residentialroad.png", 70, 70, 70, 255, true, &roadOk);
    if (!roadOk) {
        SDL_Log("Renderer: using fallback road texture.");
    }
    bool office0Ok = false;
    texOfficeFacade0 = LoadTexture2D("assets/textures/office_facade_artdeco.png", 180, 180, 180, 255, true, &office0Ok);
    if (!office0Ok) {
        SDL_Log("Renderer: using fallback office facade texture 0.");
    }
    bool office1Ok = false;
    texOfficeFacade1 = LoadTexture2D("assets/textures/office_facade_modern1.png", 180, 180, 180, 255, true, &office1Ok);
    if (!office1Ok) {
        SDL_Log("Renderer: using fallback office facade texture 1.");
    }
    bool office2Ok = false;
    texOfficeFacade2 = LoadTexture2D("assets/textures/office_facade_modern2.png", 180, 180, 180, 255, true, &office2Ok);
    if (!office2Ok) {
        SDL_Log("Renderer: using fallback office facade texture 2.");
    }
    bool office3Ok = false;
    texOfficeFacade3 = LoadTexture2D("assets/textures/office_facade_modern3.png", 180, 180, 180, 255, true, &office3Ok);
    if (!office3Ok) {
        SDL_Log("Renderer: using fallback office facade texture 3.");
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
    texSkybox = LoadCubemap(skyFaces, true, &skyOk);
    if (!skyOk) {
        SDL_Log("Renderer: using fallback skybox texture.");
    }

    // Ground quad
    const float HALF = MAP_HALF_M;
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RoadVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(RoadVertex), (void*)(sizeof(glm::vec3)));
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoPreview);
    glGenBuffers(1, &vboPreview);
    glBindVertexArray(vaoPreview);
    glBindBuffer(GL_ARRAY_BUFFER, vboPreview);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoWater);
    glGenBuffers(1, &vboWater);
    glBindVertexArray(vaoWater);
    glBindBuffer(GL_ARRAY_BUFFER, vboWater);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    // Cube mesh
    const VertexPN cube[36] = {
        {{-0.5f,-0.5f, 0.5f},{ 0.0f, 0.0f, 1.0f}}, {{ 0.5f,-0.5f, 0.5f},{ 0.0f, 0.0f, 1.0f}}, {{ 0.5f, 0.5f, 0.5f},{ 0.0f, 0.0f, 1.0f}},
        {{-0.5f,-0.5f, 0.5f},{ 0.0f, 0.0f, 1.0f}}, {{ 0.5f, 0.5f, 0.5f},{ 0.0f, 0.0f, 1.0f}}, {{-0.5f, 0.5f, 0.5f},{ 0.0f, 0.0f, 1.0f}},
        {{ 0.5f,-0.5f,-0.5f},{ 0.0f, 0.0f,-1.0f}}, {{-0.5f,-0.5f,-0.5f},{ 0.0f, 0.0f,-1.0f}}, {{-0.5f, 0.5f,-0.5f},{ 0.0f, 0.0f,-1.0f}},
        {{ 0.5f,-0.5f,-0.5f},{ 0.0f, 0.0f,-1.0f}}, {{-0.5f, 0.5f,-0.5f},{ 0.0f, 0.0f,-1.0f}}, {{ 0.5f, 0.5f,-0.5f},{ 0.0f, 0.0f,-1.0f}},
        {{ 0.5f,-0.5f, 0.5f},{ 1.0f, 0.0f, 0.0f}}, {{ 0.5f,-0.5f,-0.5f},{ 1.0f, 0.0f, 0.0f}}, {{ 0.5f, 0.5f,-0.5f},{ 1.0f, 0.0f, 0.0f}},
        {{ 0.5f,-0.5f, 0.5f},{ 1.0f, 0.0f, 0.0f}}, {{ 0.5f, 0.5f,-0.5f},{ 1.0f, 0.0f, 0.0f}}, {{ 0.5f, 0.5f, 0.5f},{ 1.0f, 0.0f, 0.0f}},
        {{-0.5f,-0.5f,-0.5f},{-1.0f, 0.0f, 0.0f}}, {{-0.5f,-0.5f, 0.5f},{-1.0f, 0.0f, 0.0f}}, {{-0.5f, 0.5f, 0.5f},{-1.0f, 0.0f, 0.0f}},
        {{-0.5f,-0.5f,-0.5f},{-1.0f, 0.0f, 0.0f}}, {{-0.5f, 0.5f, 0.5f},{-1.0f, 0.0f, 0.0f}}, {{-0.5f, 0.5f,-0.5f},{-1.0f, 0.0f, 0.0f}},
        {{-0.5f, 0.5f, 0.5f},{ 0.0f, 1.0f, 0.0f}}, {{ 0.5f, 0.5f, 0.5f},{ 0.0f, 1.0f, 0.0f}}, {{ 0.5f, 0.5f,-0.5f},{ 0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f, 0.5f},{ 0.0f, 1.0f, 0.0f}}, {{ 0.5f, 0.5f,-0.5f},{ 0.0f, 1.0f, 0.0f}}, {{-0.5f, 0.5f,-0.5f},{ 0.0f, 1.0f, 0.0f}},
        {{-0.5f,-0.5f,-0.5f},{ 0.0f,-1.0f, 0.0f}}, {{ 0.5f,-0.5f,-0.5f},{ 0.0f,-1.0f, 0.0f}}, {{ 0.5f,-0.5f, 0.5f},{ 0.0f,-1.0f, 0.0f}},
        {{-0.5f,-0.5f,-0.5f},{ 0.0f,-1.0f, 0.0f}}, {{ 0.5f,-0.5f, 0.5f},{ 0.0f,-1.0f, 0.0f}}, {{-0.5f,-0.5f, 0.5f},{ 0.0f,-1.0f, 0.0f}},
    };

    glGenBuffers(1, &vboCube);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube), cube, GL_STATIC_DRAW);

    glGenVertexArrays(1, &vaoSkybox);
    glBindVertexArray(vaoSkybox);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), (void*)0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoCubeSingle);
    glBindVertexArray(vaoCubeSingle);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), (void*)(sizeof(glm::vec3)));
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoCubeInstAnim);
    glGenBuffers(1, &vboInstAnim);

    glBindVertexArray(vaoCubeInstAnim);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), (void*)(sizeof(glm::vec3)));
    glBindBuffer(GL_ARRAY_BUFFER, vboInstAnim);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    SetupInstanceAttribs(vaoCubeInstAnim, vboInstAnim);

    glBindVertexArray(0);
    return true;
}

void Renderer::resize(int w, int h) {
    viewportW = w;
    viewportH = h;
    glViewport(0, 0, w, h);
}

void Renderer::updateRoadMesh(const std::vector<RoadVertex>& verts) {
    UploadDynamicRoadVerts(vboRoad, capRoad, verts);
}

void Renderer::updateWaterMesh(const std::vector<glm::vec3>& verts) {
    UploadDynamicVerts(vboWater, capWater, verts);
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
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(glm::vec3)));
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
    float shadowStrength = (shadowTex && shadowFbo && frame.lighting.sunIntensity > 0.001f)
        ? frame.lighting.shadowStrength
        : 0.0f;

    if (shadowStrength > 0.0f) {
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
        glViewport(0, 0, shadowMapSize, shadowMapSize);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.0f, 4.0f);

        glUseProgram(progDepthInst);
        glUniformMatrix4fv(locLightVP_DI, 1, GL_FALSE, &frame.lightViewProj[0][0]);

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
        glDisable(GL_POLYGON_OFFSET_FILL);
        glCullFace(GL_BACK);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, viewportW, viewportH);
    }

    glClearColor(0.55f, 0.75f, 0.95f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Ground/roads/overlays are single-sided quads; render without culling
    glDisable(GL_CULL_FACE);

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(progSky);
    glUniformMatrix4fv(locVP_S, 1, GL_FALSE, &frame.viewProjSky[0][0]);
    glUniform1f(locSkyBright_S, frame.lighting.skyBrightness);
    glUniform1f(locExposure_S, frame.lighting.exposure);
    glUniform1f(locSkyExposure_S, frame.lighting.skyExposure);
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
    glUniformMatrix4fv(locLightVP_G, 1, GL_FALSE, &frame.lightViewProj[0][0]);
    glUniform3f(locSunDir_G, frame.lighting.sunDir.x, frame.lighting.sunDir.y, frame.lighting.sunDir.z);
    glUniform3f(locSunColor_G, frame.lighting.sunColor.x, frame.lighting.sunColor.y, frame.lighting.sunColor.z);
    glUniform1f(locSunInt_G, frame.lighting.sunIntensity);
    glUniform3f(locAmbColor_G, frame.lighting.ambientColor.x, frame.lighting.ambientColor.y, frame.lighting.ambientColor.z);
    glUniform1f(locAmbInt_G, frame.lighting.ambientIntensity);
    glUniform1f(locExposure_G, frame.lighting.exposure);
    glUniform1f(locShadowStrength_G, shadowStrength);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, shadowTex);
    glUniform1i(locShadowMap_G, 2);
    glUniform2f(locShadowTexel_G, 1.0f / (float)shadowMapSize, 1.0f / (float)shadowMapSize);
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

    if (frame.waterVertexCount > 0) {
        glUniform1f(locGrassTile_G, 8.0f);
        glUniform1f(locNoiseTile_G, 64.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texWater);
        glUniform1i(locGrassTex_G, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texNoise);
        glUniform1i(locNoiseTex_G, 1);
        glBindVertexArray(vaoWater);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)frame.waterVertexCount);
    }

    if (frame.roadVertexCount > 0) {
        glUseProgram(progRoad);
        glUniformMatrix4fv(locVP_R, 1, GL_FALSE, &frame.viewProj[0][0]);
        glUniformMatrix4fv(locLightVP_R, 1, GL_FALSE, &frame.lightViewProj[0][0]);
        glUniform3f(locSunDir_R, frame.lighting.sunDir.x, frame.lighting.sunDir.y, frame.lighting.sunDir.z);
        glUniform3f(locSunColor_R, frame.lighting.sunColor.x, frame.lighting.sunColor.y, frame.lighting.sunColor.z);
        glUniform1f(locSunInt_R, frame.lighting.sunIntensity);
        glUniform3f(locAmbColor_R, frame.lighting.ambientColor.x, frame.lighting.ambientColor.y, frame.lighting.ambientColor.z);
        glUniform1f(locAmbInt_R, frame.lighting.ambientIntensity);
        glUniform1f(locExposure_R, frame.lighting.exposure);
        glUniform1f(locShadowStrength_R, shadowStrength);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texRoad);
        glUniform1i(locRoadTex_R, 0);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, shadowTex);
        glUniform1i(locShadowMap_R, 2);
        glUniform2f(locShadowTexel_R, 1.0f / (float)shadowMapSize, 1.0f / (float)shadowMapSize);
        glBindVertexArray(vaoRoad);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)frame.roadVertexCount);
    }

    glUseProgram(progBasic);
    glUniformMatrix4fv(locVP_B, 1, GL_FALSE, &frame.viewProj[0][0]);
    glUniformMatrix4fv(locM_B, 1, GL_FALSE, &I[0][0]);
    glUniform1f(locExposure_B, frame.lighting.exposure);

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
    glUniformMatrix4fv(locLightVP_I, 1, GL_FALSE, &frame.lightViewProj[0][0]);
    glUniform3f(locSunDir_I, frame.lighting.sunDir.x, frame.lighting.sunDir.y, frame.lighting.sunDir.z);
    glUniform3f(locSunColor_I, frame.lighting.sunColor.x, frame.lighting.sunColor.y, frame.lighting.sunColor.z);
    glUniform1f(locSunInt_I, frame.lighting.sunIntensity);
    glUniform3f(locAmbColor_I, frame.lighting.ambientColor.x, frame.lighting.ambientColor.y, frame.lighting.ambientColor.z);
    glUniform1f(locAmbInt_I, frame.lighting.ambientIntensity);
    glUniform1f(locExposure_I, frame.lighting.exposure);
    glUniform1f(locShadowStrength_I, shadowStrength);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, shadowTex);
    glUniform1i(locShadowMap_I, 2);
    glUniform2f(locShadowTexel_I, 1.0f / (float)shadowMapSize, 1.0f / (float)shadowMapSize);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texOfficeFacade0);
    glUniform1i(locFacadeTex0_I, 3);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, texOfficeFacade1);
    glUniform1i(locFacadeTex1_I, 4);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, texOfficeFacade2);
    glUniform1i(locFacadeTex2_I, 5);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, texOfficeFacade3);
    glUniform1i(locFacadeTex3_I, 6);
    glUniform2f(locFacadeTile_I, 8.0f, 4.0f);
    glUniform3f(locFacadeTint_I, 1.0f, 1.0f, 1.0f);
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
    if (progRoad) { glDeleteProgram(progRoad); progRoad = 0; }
    if (progSky) { glDeleteProgram(progSky); progSky = 0; }
    if (progDepth) { glDeleteProgram(progDepth); progDepth = 0; }
    if (progDepthInst) { glDeleteProgram(progDepthInst); progDepthInst = 0; }
    if (texGrass) { glDeleteTextures(1, &texGrass); texGrass = 0; }
    if (texNoise) { glDeleteTextures(1, &texNoise); texNoise = 0; }
    if (texWater) { glDeleteTextures(1, &texWater); texWater = 0; }
    if (texRoad) { glDeleteTextures(1, &texRoad); texRoad = 0; }
    if (texOfficeFacade0) { glDeleteTextures(1, &texOfficeFacade0); texOfficeFacade0 = 0; }
    if (texOfficeFacade1) { glDeleteTextures(1, &texOfficeFacade1); texOfficeFacade1 = 0; }
    if (texOfficeFacade2) { glDeleteTextures(1, &texOfficeFacade2); texOfficeFacade2 = 0; }
    if (texOfficeFacade3) { glDeleteTextures(1, &texOfficeFacade3); texOfficeFacade3 = 0; }
    if (texSkybox) { glDeleteTextures(1, &texSkybox); texSkybox = 0; }
    if (shadowTex) { glDeleteTextures(1, &shadowTex); shadowTex = 0; }
    if (shadowFbo) { glDeleteFramebuffers(1, &shadowFbo); shadowFbo = 0; }

    GLuint vaos[] = { vaoGround, vaoRoad, vaoPreview, vaoSkybox, vaoWater, vaoCubeSingle, vaoCubeInstAnim };
    GLuint vbos[] = { vboGround, vboRoad, vboPreview, vboWater, vboCube, vboInstAnim };

    glDeleteVertexArrays((GLsizei)std::size(vaos), vaos);
    glDeleteBuffers((GLsizei)std::size(vbos), vbos);

    for (auto& chunkKv : houseChunks) {
        for (auto& assetKv : chunkKv.second) {
            if (assetKv.second.vao) glDeleteVertexArrays(1, &assetKv.second.vao);
            if (assetKv.second.vbo) glDeleteBuffers(1, &assetKv.second.vbo);
        }
    }
    houseChunks.clear();

    vaoGround = vaoRoad = vaoPreview = vaoSkybox = vaoWater = vaoCubeSingle = vaoCubeInstAnim = 0;
    vboGround = vboRoad = vboPreview = vboWater = vboCube = vboInstAnim = 0;
}

void Renderer::shutdown() {
    destroyGL();
}
