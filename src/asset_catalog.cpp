#include "asset_catalog.h"

#include <SDL.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

using json = nlohmann::json;

namespace {

glm::vec2 ParseVec2(const json& j, const char* key, const glm::vec2& fallback) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_array() || it->size() != 2) return fallback;
    return glm::vec2((*it)[0].get<float>(), (*it)[1].get<float>());
}

glm::vec3 ParseVec3(const json& j, const char* key, const glm::vec3& fallback) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_array() || it->size() != 3) return fallback;
    return glm::vec3((*it)[0].get<float>(), (*it)[1].get<float>(), (*it)[2].get<float>());
}

bool HasRequiredFields(const json& j) {
    return j.contains("version") && j.contains("id") && j.contains("type") && j.contains("mesh");
}

} // namespace

AssetId AssetCatalog::HashId(const std::string& idStr) {
    const uint32_t fnvOffset = 2166136261u;
    const uint32_t fnvPrime = 16777619u;
    uint32_t hash = fnvOffset;
    for (unsigned char c : idStr) {
        hash ^= c;
        hash *= fnvPrime;
    }
    return hash;
}

bool AssetCatalog::registerAsset(AssetDef def) {
    if (def.idStr.empty()) return false;
    if (def.id == 0) def.id = HashId(def.idStr);
    if (assetsById.find(def.id) != assetsById.end()) return false;
    assetsById[def.id] = def;
    assetsByStr[def.idStr] = def.id;
    return true;
}

void AssetCatalog::registerBuiltinDefaults() {
    defaultByCategoryStr["low_density"] = "buildings.house_low_01";

    AssetDef fallback;
    fallback.idStr = "builtin.cube_house";
    fallback.id = HashId(fallback.idStr);
    fallback.type = "building";
    fallback.category = "fallback";
    fallback.meshRelPath = "";
    fallback.defaultScale = glm::vec3(1.0f, 1.0f, 1.0f);
    fallback.footprintM = glm::vec2(1.0f, 1.0f);
    fallback.pivotM = glm::vec3(0.0f, 0.0f, 0.0f);
    fallback.tags = {"fallback"};
    registerAsset(fallback);
    fallbackId = fallback.id;
}

bool AssetCatalog::loadAll(const std::string& assetsRoot) {
    assetsById.clear();
    assetsByStr.clear();
    defaultByCategoryStr.clear();
    fallbackId = 0;
    rootPath = assetsRoot;

    registerBuiltinDefaults();

    std::filesystem::path root(assetsRoot);
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        SDL_Log("AssetCatalog: assets root not found: %s", assetsRoot.c_str());
        return false;
    }

    bool loadedAny = false;
    for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        if (ec) {
            SDL_Log("AssetCatalog: error scanning assets: %s", ec.message().c_str());
            break;
        }
        if (!it->is_regular_file(ec)) continue;
        if (it->path().filename() != "asset.json") continue;

        std::ifstream in(it->path(), std::ios::binary);
        if (!in) {
            SDL_Log("AssetCatalog: failed to open %s", it->path().string().c_str());
            continue;
        }

        json j;
        try {
            in >> j;
        } catch (const std::exception& e) {
            SDL_Log("AssetCatalog: failed to parse %s (%s)", it->path().string().c_str(), e.what());
            continue;
        }

        if (!HasRequiredFields(j)) {
            SDL_Log("AssetCatalog: missing fields in %s", it->path().string().c_str());
            continue;
        }

        AssetDef def;
        def.idStr = j.value("id", "");
        def.id = HashId(def.idStr);
        def.type = j.value("type", "");
        def.category = j.value("category", "");
        def.meshRelPath = j.value("mesh", "");
        def.defaultScale = ParseVec3(j, "defaultScale", def.defaultScale);
        def.footprintM = ParseVec2(j, "footprintM", def.footprintM);
        def.pivotM = ParseVec3(j, "pivotM", def.pivotM);

        auto tagsIt = j.find("tags");
        if (tagsIt != j.end() && tagsIt->is_array()) {
            for (const auto& t : *tagsIt) {
                if (t.is_string()) def.tags.push_back(t.get<std::string>());
            }
        }

        if (!registerAsset(def)) {
            SDL_Log("AssetCatalog: duplicate asset id %s (%s)", def.idStr.c_str(), it->path().string().c_str());
            continue;
        }

        loadedAny = true;
    }

    return loadedAny;
}

const AssetDef* AssetCatalog::find(AssetId id) const {
    auto it = assetsById.find(id);
    if (it == assetsById.end()) return nullptr;
    return &it->second;
}

AssetId AssetCatalog::findIdByString(const std::string& idStr) const {
    auto it = assetsByStr.find(idStr);
    if (it == assetsByStr.end()) return 0;
    return it->second;
}

AssetId AssetCatalog::resolveCategoryAsset(const std::string& category) const {
    auto it = defaultByCategoryStr.find(category);
    if (it != defaultByCategoryStr.end()) {
        AssetId id = HashId(it->second);
        if (assetsById.find(id) != assetsById.end()) return id;
    }
    return fallbackId;
}
