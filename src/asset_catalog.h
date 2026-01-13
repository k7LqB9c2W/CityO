#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using AssetId = uint32_t;

struct AssetDef {
    std::string idStr;
    AssetId id = 0;
    std::string type;
    std::string category;
    std::string meshRelPath;
    glm::vec3 defaultScale{1.0f, 1.0f, 1.0f};
    glm::vec2 footprintM{0.0f, 0.0f};
    glm::vec2 zonedFootprintM{0.0f, 0.0f};
    glm::vec3 pivotM{0.0f, 0.0f, 0.0f};
    std::vector<std::string> tags;
};

class AssetCatalog {
public:
    bool loadAll(const std::string& assetsRoot);
    const AssetDef* find(AssetId id) const;
    AssetId findIdByString(const std::string& idStr) const;
    AssetId resolveCategoryAsset(const std::string& category) const;
    AssetId fallbackAsset() const { return fallbackId; }
    const std::unordered_map<AssetId, AssetDef>& assets() const { return assetsById; }
    const std::string& root() const { return rootPath; }

    static AssetId HashId(const std::string& idStr);

private:
    bool registerAsset(AssetDef def);
    void registerBuiltinDefaults();

    std::string rootPath;
    std::unordered_map<AssetId, AssetDef> assetsById;
    std::unordered_map<std::string, AssetId> assetsByStr;
    std::unordered_map<std::string, std::string> defaultByCategoryStr;
    AssetId fallbackId = 0;
};
