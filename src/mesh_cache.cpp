#include "mesh_cache.h"

#include <SDL.h>
#include <filesystem>
#include <vector>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

namespace {

std::string JoinPath(const std::string& root, const std::string& rel) {
    if (root.empty()) return rel;
    std::filesystem::path full = std::filesystem::path(root) / rel;
    return full.string();
}

struct VertexPN {
    glm::vec3 pos;
    glm::vec3 normal;
};

} // namespace

bool MeshCache::init() {
    buildFallbackCube();
    return fallback.vbo != 0;
}

void MeshCache::destroyMesh(MeshGpu& mesh) {
    if (mesh.vbo) glDeleteBuffers(1, &mesh.vbo);
    if (mesh.ebo) glDeleteBuffers(1, &mesh.ebo);
    mesh = MeshGpu{};
}

void MeshCache::shutdown() {
    for (auto& kv : loaded) destroyMesh(kv.second);
    loaded.clear();
    failed.clear();
    destroyMesh(fallback);
}

const MeshGpu& MeshCache::getOrLoad(AssetId assetId, const AssetCatalog& catalog) {
    auto it = loaded.find(assetId);
    if (it != loaded.end()) return it->second;
    if (failed.find(assetId) != failed.end()) return fallback;

    const AssetDef* def = catalog.find(assetId);
    if (!def || def->meshRelPath.empty()) {
        failed.insert(assetId);
        return fallback;
    }

    if (!loadMeshForAsset(assetId, *def, catalog.root())) {
        failed.insert(assetId);
        return fallback;
    }

    return loaded[assetId];
}

bool MeshCache::loadMeshForAsset(AssetId assetId, const AssetDef& def, const std::string& root) {
    MeshGpu mesh;
    std::string path = JoinPath(root, def.meshRelPath);
    if (!loadGltfMesh(path, mesh)) return false;
    loaded.emplace(assetId, mesh);
    return true;
}

bool MeshCache::loadGltfMesh(const std::string& path, MeshGpu& out) {
    cgltf_options options{};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) {
        SDL_Log("MeshCache: cgltf_parse_file failed: %s", path.c_str());
        return false;
    }

    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success) {
        SDL_Log("MeshCache: cgltf_load_buffers failed: %s", path.c_str());
        cgltf_free(data);
        return false;
    }

    if (data->meshes_count == 0 || data->meshes == nullptr) {
        SDL_Log("MeshCache: no meshes in %s", path.c_str());
        cgltf_free(data);
        return false;
    }

    const cgltf_mesh* mesh = &data->meshes[0];
    if (mesh->primitives_count == 0 || mesh->primitives == nullptr) {
        SDL_Log("MeshCache: no primitives in %s", path.c_str());
        cgltf_free(data);
        return false;
    }

    const cgltf_primitive* prim = &mesh->primitives[0];
    const cgltf_accessor* posAcc = nullptr;
    const cgltf_accessor* normAcc = nullptr;
    for (size_t i = 0; i < prim->attributes_count; i++) {
        const cgltf_attribute& attr = prim->attributes[i];
        if (attr.type == cgltf_attribute_type_position) {
            posAcc = attr.data;
        } else if (attr.type == cgltf_attribute_type_normal) {
            normAcc = attr.data;
        }
    }

    if (!posAcc || posAcc->type != cgltf_type_vec3) {
        SDL_Log("MeshCache: missing POSITION attribute in %s", path.c_str());
        cgltf_free(data);
        return false;
    }

    const size_t vertCount = posAcc->count;
    std::vector<glm::vec3> positions(vertCount);
    for (size_t i = 0; i < vertCount; i++) {
        float v[3] = {0.0f, 0.0f, 0.0f};
        cgltf_accessor_read_float(posAcc, i, v, 3);
        positions[i] = glm::vec3(v[0], v[1], v[2]);
    }

    std::vector<uint32_t> indices;
    bool indexed = prim->indices != nullptr;
    if (indexed) {
        const cgltf_accessor* idxAcc = prim->indices;
        indices.resize(idxAcc->count);
        for (size_t i = 0; i < idxAcc->count; i++) {
            indices[i] = (uint32_t)cgltf_accessor_read_index(idxAcc, i);
        }
    }

    std::vector<glm::vec3> normals(vertCount, glm::vec3(0.0f));
    bool hasNormals = (normAcc && normAcc->type == cgltf_type_vec3 && normAcc->count == vertCount);
    if (hasNormals) {
        for (size_t i = 0; i < vertCount; i++) {
            float v[3] = {0.0f, 1.0f, 0.0f};
            cgltf_accessor_read_float(normAcc, i, v, 3);
            normals[i] = glm::vec3(v[0], v[1], v[2]);
        }
    } else {
        if (indexed && indices.size() >= 3) {
            size_t triCount = indices.size() / 3;
            for (size_t t = 0; t < triCount; t++) {
                uint32_t i0 = indices[t * 3 + 0];
                uint32_t i1 = indices[t * 3 + 1];
                uint32_t i2 = indices[t * 3 + 2];
                if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
                glm::vec3 e0 = positions[i1] - positions[i0];
                glm::vec3 e1 = positions[i2] - positions[i0];
                glm::vec3 n = glm::cross(e0, e1);
                float len = glm::length(n);
                if (len > 1e-6f) {
                    n /= len;
                    normals[i0] += n;
                    normals[i1] += n;
                    normals[i2] += n;
                }
            }
            for (size_t i = 0; i < vertCount; i++) {
                float len = glm::length(normals[i]);
                normals[i] = (len > 1e-6f) ? normals[i] / len : glm::vec3(0.0f, 1.0f, 0.0f);
            }
        } else {
            for (size_t i = 0; i + 2 < vertCount; i += 3) {
                glm::vec3 e0 = positions[i + 1] - positions[i];
                glm::vec3 e1 = positions[i + 2] - positions[i];
                glm::vec3 n = glm::cross(e0, e1);
                float len = glm::length(n);
                n = (len > 1e-6f) ? (n / len) : glm::vec3(0.0f, 1.0f, 0.0f);
                normals[i] = n;
                normals[i + 1] = n;
                normals[i + 2] = n;
            }
        }
    }

    cgltf_free(data);

    std::vector<VertexPN> vertices(vertCount);
    for (size_t i = 0; i < vertCount; i++) {
        vertices[i].pos = positions[i];
        vertices[i].normal = normals[i];
    }

    MeshGpu meshOut;
    glGenBuffers(1, &meshOut.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, meshOut.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vertices.size() * sizeof(VertexPN)), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (indexed && !indices.empty()) {
        glGenBuffers(1, &meshOut.ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshOut.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(indices.size() * sizeof(uint32_t)), indices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        meshOut.indexed = true;
        meshOut.indexCount = (GLsizei)indices.size();
    } else {
        meshOut.indexed = false;
        meshOut.vertexCount = (GLsizei)vertCount;
    }

    meshOut.vertexStride = (GLsizei)sizeof(VertexPN);
    meshOut.vertexCount = (GLsizei)vertCount;
    out = meshOut;
    return true;
}

void MeshCache::buildFallbackCube() {
    static const VertexPN cubeVerts[] = {
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

    glGenBuffers(1, &fallback.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, fallback.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    fallback.vertexStride = (GLsizei)sizeof(VertexPN);
    fallback.vertexCount = 36;
    fallback.indexCount = 0;
    fallback.indexed = false;
}
