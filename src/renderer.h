#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>

#include "asset_catalog.h"
#include "lighting.h"

struct RenderMarker {
    glm::vec3 pos{};
    glm::vec3 color{};
    float scale = 1.0f;
};

struct RenderHouseBatch {
    uint64_t chunkKey = 0;
    AssetId asset = 0;
};

struct RoadVertex {
    glm::vec3 pos{};
    glm::vec2 uv{};
};

struct RenderFrame {
    glm::mat4 viewProj{1.0f};
    glm::mat4 viewProjSky{1.0f};
    glm::mat4 lightViewProj{1.0f};
    glm::vec3 cameraPos{0.0f};
    glm::vec3 cameraTarget{0.0f};
    LightingParams lighting;
    std::size_t roadVertexCount = 0;
    std::size_t waterVertexCount = 0;
    std::size_t gridVertexCount = 0;
    std::size_t zoneResidentialVertexCount = 0;
    std::size_t zoneCommercialVertexCount = 0;
    std::size_t zoneIndustrialVertexCount = 0;
    std::size_t zoneOfficeVertexCount = 0;
    std::size_t previewVertexCount = 0;
    bool drawRoadPreview = false;
    bool zonePreviewValid = true;
    uint8_t zonePreviewType = 0;
    std::vector<RenderMarker> markers;
    std::vector<RenderHouseBatch> visibleHouseBatches;
    std::size_t houseAnimCount = 0;
};

struct HouseInstanceGPU {
    glm::vec4 posYaw;    // xyz position, w = yaw (radians)
    glm::vec4 scaleVar;  // xyz scale, w = variant/unused
};

struct MeshGpu;

class Renderer {
public:
    bool init();
    void resize(int w, int h);
    void updateRoadMesh(const std::vector<RoadVertex>& verts);
    void updateWaterMesh(const std::vector<glm::vec3>& verts);
    void updatePreviewMesh(const std::vector<glm::vec3>& verts);
    void updateHouseChunk(uint64_t key, AssetId assetId, const MeshGpu& mesh, const std::vector<HouseInstanceGPU>& instances);
    void updateAnimHouses(const std::vector<HouseInstanceGPU>& animHouses);
    void render(const RenderFrame& frame);
    void shutdown();

private:
    void destroyGL();

    // Programs
    unsigned int progBasic = 0;
    unsigned int progInst = 0;
    unsigned int progGround = 0;
    unsigned int progRoad = 0;
    unsigned int progSky = 0;
    unsigned int progDepth = 0;
    unsigned int progDepthInst = 0;

    // Uniform locations
    int locVP_B = -1;
    int locM_B = -1;
    int locC_B = -1;
    int locA_B = -1;
    int locExposure_B = -1;
    int locVP_I = -1;
    int locC_I = -1;
    int locA_I = -1;
    int locSunDir_I = -1;
    int locSunColor_I = -1;
    int locSunInt_I = -1;
    int locAmbColor_I = -1;
    int locAmbInt_I = -1;
    int locExposure_I = -1;
    int locLightVP_I = -1;
    int locShadowMap_I = -1;
    int locShadowTexel_I = -1;
    int locShadowStrength_I = -1;
    int locVP_G = -1;
    int locM_G = -1;
    int locGrassTile_G = -1;
    int locNoiseTile_G = -1;
    int locGrassTex_G = -1;
    int locNoiseTex_G = -1;
    int locSunDir_G = -1;
    int locSunColor_G = -1;
    int locSunInt_G = -1;
    int locAmbColor_G = -1;
    int locAmbInt_G = -1;
    int locExposure_G = -1;
    int locLightVP_G = -1;
    int locShadowMap_G = -1;
    int locShadowTexel_G = -1;
    int locShadowStrength_G = -1;
    int locVP_R = -1;
    int locLightVP_R = -1;
    int locRoadTex_R = -1;
    int locSunDir_R = -1;
    int locSunColor_R = -1;
    int locSunInt_R = -1;
    int locAmbColor_R = -1;
    int locAmbInt_R = -1;
    int locExposure_R = -1;
    int locShadowMap_R = -1;
    int locShadowTexel_R = -1;
    int locShadowStrength_R = -1;
    int locVP_S = -1;
    int locSkyTex_S = -1;
    int locSkyBright_S = -1;
    int locExposure_S = -1;
    int locLightVP_D = -1;
    int locM_D = -1;
    int locLightVP_DI = -1;

    // Buffers / VAOs
    unsigned int vaoGround = 0;
    unsigned int vboGround = 0;
    unsigned int texGrass = 0;
    unsigned int texNoise = 0;
    unsigned int texWater = 0;
    unsigned int texRoad = 0;
    unsigned int vaoSkybox = 0;
    unsigned int texSkybox = 0;
    unsigned int vaoWater = 0;
    unsigned int vboWater = 0;

    unsigned int vaoRoad = 0;
    unsigned int vboRoad = 0;

    unsigned int vaoPreview = 0;
    unsigned int vboPreview = 0;

    unsigned int vboCube = 0;
    unsigned int vaoCubeSingle = 0;

    unsigned int vaoCubeInstAnim = 0;
    unsigned int vboInstAnim = 0;

    unsigned int shadowFbo = 0;
    unsigned int shadowTex = 0;
    int shadowMapSize = 2048;

    int viewportW = 0;
    int viewportH = 0;

    struct ChunkBuf {
        unsigned int vao = 0;
        unsigned int vbo = 0;
        unsigned int meshVbo = 0;
        unsigned int meshEbo = 0;
        std::size_t vertexCount = 0;
        std::size_t indexCount = 0;
        bool indexed = false;
        std::size_t count = 0;
        std::size_t capacity = 0;
    };
    std::unordered_map<uint64_t, std::unordered_map<AssetId, ChunkBuf>> houseChunks;

    // Buffer capacities to avoid reallocation thrash
    std::size_t capRoad = 0;
    std::size_t capWater = 0;
    std::size_t capPreview = 0;
    std::size_t capInstAnim = 0;
};
