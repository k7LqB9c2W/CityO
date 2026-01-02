#pragma once

#include <glm/glm.hpp>
#include <vector>

struct RenderMarker {
    glm::vec3 pos{};
    glm::vec3 color{};
    float scale = 1.0f;
};

struct RenderFrame {
    glm::mat4 viewProj{1.0f};
    std::size_t roadVertexCount = 0;
    std::size_t overlayVertexCount = 0;
    std::size_t previewVertexCount = 0;
    std::size_t houseStaticCount = 0;
    std::size_t houseAnimCount = 0;
    bool drawRoadPreview = false;
    bool zonePreviewValid = true;
    std::vector<RenderMarker> markers;
};

class Renderer {
public:
    bool init();
    void resize(int w, int h);
    void updateRoadMesh(const std::vector<glm::vec3>& verts);
    void updatePreviewMesh(const std::vector<glm::vec3>& verts);
    void updateHouseInstances(const std::vector<glm::mat4>& staticHouses,
                              const std::vector<glm::mat4>& animHouses);
    void render(const RenderFrame& frame);
    void shutdown();

private:
    void destroyGL();

    // Programs
    unsigned int progBasic = 0;
    unsigned int progInst = 0;

    // Uniform locations
    int locVP_B = -1;
    int locM_B = -1;
    int locC_B = -1;
    int locA_B = -1;
    int locVP_I = -1;
    int locC_I = -1;
    int locA_I = -1;

    // Buffers / VAOs
    unsigned int vaoGround = 0;
    unsigned int vboGround = 0;

    unsigned int vaoRoad = 0;
    unsigned int vboRoad = 0;

    unsigned int vaoPreview = 0;
    unsigned int vboPreview = 0;

    unsigned int vboCube = 0;
    unsigned int vaoCubeSingle = 0;

    unsigned int vaoCubeInstStatic = 0;
    unsigned int vaoCubeInstAnim = 0;
    unsigned int vboInstStatic = 0;
    unsigned int vboInstAnim = 0;
};
