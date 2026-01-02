#include "renderer.h"

#include <SDL.h>
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <string>

namespace {

GLuint MakeProgram(const char* vsSrc, const char* fsSrc);
void GLCheckShader(GLuint shader, const char* label);
void GLCheckProgram(GLuint prog);
void SetupInstanceAttribs(GLuint vao, GLuint instanceVbo);
void UploadDynamicVerts(GLuint vbo, const std::vector<glm::vec3>& verts);

void GLCheckShader(GLuint shader, const char* label) {
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string log((size_t)len, '\0');
        glGetShaderInfoLog(shader, len, &len, log.data());
        SDL_Log("Shader compile failed (%s): %s", label, log.c_str());
    }
}

void GLCheckProgram(GLuint prog) {
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log((size_t)len, '\0');
        glGetProgramInfoLog(prog, len, &len, log.data());
        SDL_Log("Program link failed: %s", log.c_str());
    }
}

GLuint MakeProgram(const char* vsSrc, const char* fsSrc) {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSrc, nullptr);
    glCompileShader(vs);
    GLCheckShader(vs, "VS");

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, nullptr);
    glCompileShader(fs);
    GLCheckShader(fs, "FS");

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLCheckProgram(prog);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void SetupInstanceAttribs(GLuint vao, GLuint instanceVbo) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);

    std::size_t vec4Size = sizeof(glm::vec4);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(0 * vec4Size));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(1 * vec4Size));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * vec4Size));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * vec4Size));

    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);

    glBindVertexArray(0);
}

void UploadDynamicVerts(GLuint vbo, const std::vector<glm::vec3>& verts) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    if (!verts.empty()) {
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(glm::vec3)), verts.data(), GL_DYNAMIC_DRAW);
    } else {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

} // namespace

bool Renderer::init() {
    glEnable(GL_DEPTH_TEST);

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
        layout(location=2) in mat4 iModel;
        uniform mat4 uViewProj;
        void main() {
            gl_Position = uViewProj * iModel * vec4(aPos, 1.0);
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

    progBasic = MakeProgram(vsBasic, fsColor);
    progInst = MakeProgram(vsInstanced, fsColor);

    locVP_B = glGetUniformLocation(progBasic, "uViewProj");
    locM_B = glGetUniformLocation(progBasic, "uModel");
    locC_B = glGetUniformLocation(progBasic, "uColor");
    locA_B = glGetUniformLocation(progBasic, "uAlpha");

    locVP_I = glGetUniformLocation(progInst, "uViewProj");
    locC_I = glGetUniformLocation(progInst, "uColor");
    locA_I = glGetUniformLocation(progInst, "uAlpha");

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

    glGenVertexArrays(1, &vaoCubeSingle);
    glBindVertexArray(vaoCubeSingle);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoCubeInstStatic);
    glGenVertexArrays(1, &vaoCubeInstAnim);
    glGenBuffers(1, &vboInstStatic);
    glGenBuffers(1, &vboInstAnim);

    glBindVertexArray(vaoCubeInstStatic);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vboInstStatic);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    SetupInstanceAttribs(vaoCubeInstStatic, vboInstStatic);

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
    UploadDynamicVerts(vboRoad, verts);
}

void Renderer::updatePreviewMesh(const std::vector<glm::vec3>& verts) {
    UploadDynamicVerts(vboPreview, verts);
}

void Renderer::updateHouseInstances(const std::vector<glm::mat4>& staticHouses,
                                    const std::vector<glm::mat4>& animHouses) {
    glBindBuffer(GL_ARRAY_BUFFER, vboInstStatic);
    if (!staticHouses.empty()) {
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(staticHouses.size() * sizeof(glm::mat4)), staticHouses.data(), GL_DYNAMIC_DRAW);
    } else {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    }

    glBindBuffer(GL_ARRAY_BUFFER, vboInstAnim);
    if (!animHouses.empty()) {
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(animHouses.size() * sizeof(glm::mat4)), animHouses.data(), GL_DYNAMIC_DRAW);
    } else {
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::render(const RenderFrame& frame) {
    glClearColor(0.55f, 0.75f, 0.95f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(progBasic);
    glUniformMatrix4fv(locVP_B, 1, GL_FALSE, &frame.viewProj[0][0]);

    glm::mat4 I(1.0f);
    glUniformMatrix4fv(locM_B, 1, GL_FALSE, &I[0][0]);
    glUniform3f(locC_B, 0.05f, 0.20f, 0.08f);
    glUniform1f(locA_B, 1.0f);
    glBindVertexArray(vaoGround);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUniform3f(locC_B, 0.18f, 0.18f, 0.18f);
    glUniform1f(locA_B, 1.0f);
    glBindVertexArray(vaoRoad);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)frame.roadVertexCount);

    // Preview overlay
    if (frame.previewVertexCount > 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (frame.drawRoadPreview) {
            glUniform3f(locC_B, 0.20f, 0.65f, 0.95f);
            glUniform1f(locA_B, 0.50f);
        } else {
            if (frame.zonePreviewValid) glUniform3f(locC_B, 0.20f, 0.85f, 0.35f);
            else glUniform3f(locC_B, 0.90f, 0.20f, 0.20f);
            glUniform1f(locA_B, 0.35f);
        }

        glBindVertexArray(vaoPreview);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)frame.previewVertexCount);
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

    if (frame.houseStaticCount > 0) {
        glBindVertexArray(vaoCubeInstStatic);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, (GLsizei)frame.houseStaticCount);
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

    GLuint vaos[] = { vaoGround, vaoRoad, vaoPreview, vaoCubeSingle, vaoCubeInstStatic, vaoCubeInstAnim };
    GLuint vbos[] = { vboGround, vboRoad, vboPreview, vboCube, vboInstStatic, vboInstAnim };

    glDeleteVertexArrays((GLsizei)std::size(vaos), vaos);
    glDeleteBuffers((GLsizei)std::size(vbos), vbos);

    vaoGround = vaoRoad = vaoPreview = vaoCubeSingle = vaoCubeInstStatic = vaoCubeInstAnim = 0;
    vboGround = vboRoad = vboPreview = vboCube = vboInstStatic = vboInstAnim = 0;
}

void Renderer::shutdown() {
    destroyGL();
}
