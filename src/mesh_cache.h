#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "asset_catalog.h"

struct MeshGpu {
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei vertexCount = 0;
    GLsizei indexCount = 0;
    GLsizei vertexStride = 0;
    bool indexed = false;
};

class MeshCache {
public:
    bool init();
    void shutdown();
    const MeshGpu& getOrLoad(AssetId assetId, const AssetCatalog& catalog);
    const MeshGpu& fallbackMesh() const { return fallback; }

private:
    bool loadMeshForAsset(AssetId assetId, const AssetDef& def, const std::string& root);
    bool loadGltfMesh(const std::string& path, MeshGpu& out);
    void destroyMesh(MeshGpu& mesh);
    void buildFallbackCube();

    std::unordered_map<AssetId, MeshGpu> loaded;
    std::unordered_set<AssetId> failed;
    MeshGpu fallback;
};
