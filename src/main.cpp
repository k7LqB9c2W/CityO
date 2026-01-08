#include <SDL.h>
#include <glad/glad.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <nlohmann/json.hpp>

#include "renderer.h"
#include "asset_catalog.h"
#include "mesh_cache.h"
#include "config.h"
#include "image_loader.h"
#include "lighting.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <array>
#include <fstream>
#include <memory>
#include <limits>
#include <unordered_set>
#include <unordered_map>

using json = nlohmann::json;

static float Clamp(float v, float a, float b) { return (v < a) ? a : (v > b) ? b : v; }

static uint32_t Hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static float LenXZ(const glm::vec3& a, const glm::vec3& b) {
    glm::vec2 d(b.x - a.x, b.z - a.z);
    return std::sqrt(d.x*d.x + d.y*d.y);
}

constexpr float ORIGIN_STEP_M = 1024.0f;
static glm::vec3 ComputeRenderOrigin(const glm::vec3& target) {
    glm::vec3 o(0.0f);
    o.x = std::floor(target.x / ORIGIN_STEP_M) * ORIGIN_STEP_M;
    o.z = std::floor(target.z / ORIGIN_STEP_M) * ORIGIN_STEP_M;
    return o;
}

constexpr float CHUNK_SIZE_M = 1024.0f;
struct ChunkCoord { int32_t cx; int32_t cz; };
static uint64_t PackChunk(int32_t cx, int32_t cz) {
    return (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz);
}
static void UnpackChunk(uint64_t key, int32_t& cx, int32_t& cz) {
    cx = (int32_t)(key >> 32);
    cz = (int32_t)(key & 0xffffffffu);
}
static ChunkCoord ChunkFromPosXZ(const glm::vec3& p) {
    return {
        (int32_t)std::floor(p.x / CHUNK_SIZE_M),
        (int32_t)std::floor(p.z / CHUNK_SIZE_M)
    };
}

struct Camera {
    glm::vec3 target{0.0f, 0.0f, 0.0f};
    float distance = 180.0f;
    float pitchDeg = 60.0f;
    float yawRad = 0.8f;

    glm::vec3 position() const {
        float pitch = glm::radians(pitchDeg);
        float y = distance * std::sin(pitch);
        float xz = distance * std::cos(pitch);
        float x = xz * std::sin(yawRad);
        float z = xz * std::cos(yawRad);
        return target + glm::vec3(x, y, z);
    }

    glm::mat4 view() const {
        return glm::lookAt(position(), target, glm::vec3(0,1,0));
    }
};

static bool ScreenToGroundHit(
    int mx, int my, int w, int h,
    const glm::mat4& view, const glm::mat4& proj,
    glm::vec3& outHit)
{
    float x = (2.0f * mx) / float(w) - 1.0f;
    float y = 1.0f - (2.0f * my) / float(h);

    glm::mat4 invProj = glm::inverse(proj);
    glm::mat4 invView = glm::inverse(view);

    glm::vec4 rayClip(x, y, -1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);

    glm::vec3 rayWorld = glm::normalize(glm::vec3(invView * rayEye));
    glm::vec3 origin = glm::vec3(invView * glm::vec4(0,0,0,1));

    if (std::fabs(rayWorld.y) < 1e-6f) return false;
    float t = -origin.y / rayWorld.y;
    if (t < 0.0f) return false;

    outHit = origin + rayWorld * t;
    outHit.y = 0.0f;
    return true;
}

static glm::vec3 SnapToGridXZ(glm::vec3 p, float grid) {
    if (grid <= 0.0f) return p;
    p.x = std::round(p.x / grid) * grid;
    p.z = std::round(p.z / grid) * grid;
    p.y = 0.0f;
    return p;
}

static glm::vec3 SnapAngle15FromPrev(const glm::vec3& prev, const glm::vec3& raw) {
    glm::vec3 d = raw - prev;
    d.y = 0.0f;
    float len = std::sqrt(d.x*d.x + d.z*d.z);
    if (len < 1e-6f) return raw;

    float ang = std::atan2(d.z, d.x);
    float step = glm::radians(15.0f);
    float snapped = std::round(ang / step) * step;

    glm::vec3 out = prev + glm::vec3(std::cos(snapped), 0.0f, std::sin(snapped)) * len;
    out.y = 0.0f;
    return out;
}

struct Road {
    int id = 0;
    std::vector<glm::vec3> pts;
    std::vector<float> cumLen;

    void rebuildCum() {
        cumLen.clear();
        cumLen.reserve(pts.size());
        float acc = 0.0f;
        if (pts.empty()) return;
        cumLen.push_back(0.0f);
        for (size_t i = 0; i + 1 < pts.size(); i++) {
            acc += LenXZ(pts[i], pts[i+1]);
            cumLen.push_back(acc);
        }
    }

    float totalLen() const {
        if (cumLen.empty()) return 0.0f;
        return cumLen.back();
    }

    glm::vec3 pointAt(float d, glm::vec3& outTan) const {
        if (pts.size() < 2 || cumLen.size() != pts.size()) {
            outTan = glm::vec3(1,0,0);
            return pts.empty() ? glm::vec3(0,0,0) : pts[0];
        }
        d = Clamp(d, 0.0f, totalLen());

        size_t i = 0;
        while (i + 1 < cumLen.size() && cumLen[i+1] < d) i++;

        glm::vec3 a = pts[i];
        glm::vec3 b = pts[i+1];
        float segLen = std::max(1e-6f, LenXZ(a, b));
        float t = (d - cumLen[i]) / segLen;

        glm::vec3 dir = b - a;
        dir.y = 0.0f;
        float l = std::sqrt(dir.x*dir.x + dir.z*dir.z);
        if (l > 1e-6f) dir /= l;
        outTan = dir;

        glm::vec3 p = a + (b - a) * t;
        p.y = 0.0f;
        return p;
    }
};

// Forward declaration used in house placement clearance checks
static float ClosestDistanceAlongRoadSq(const Road& r, const glm::vec3& p, float& outAlong, glm::vec3& outTan);

static int FindRoadIndexById(const std::vector<Road>& roads, int id) {
    for (int i = 0; i < (int)roads.size(); i++) if (roads[i].id == id) return i;
    return -1;
}

static float ClosestParamOnSegmentXZ(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, glm::vec3& outClosest) {
    glm::vec2 ap(p.x - a.x, p.z - a.z);
    glm::vec2 ab(b.x - a.x, b.z - a.z);
    float ab2 = ab.x*ab.x + ab.y*ab.y;
    float t = (ab2 > 1e-8f) ? (ap.x*ab.x + ap.y*ab.y) / ab2 : 0.0f;
    t = Clamp(t, 0.0f, 1.0f);
    outClosest = a + (b - a) * t;
    outClosest.y = 0.0f;
    return t;
}

// Returns bestDistSq, and fills out roadId, pointIndex, isEndpoint, endpointIsStart
static bool PickRoadPoint(
    const std::vector<Road>& roads, const glm::vec3& p, float radius,
    int& outRoadId, int& outPointIndex)
{
    float bestSq = radius * radius;
    int bestRoad = -1;
    int bestPt = -1;

    for (const auto& r : roads) {
        for (int i = 0; i < (int)r.pts.size(); i++) {
            glm::vec2 d(p.x - r.pts[i].x, p.z - r.pts[i].z);
            float dsq = d.x*d.x + d.y*d.y;
            if (dsq < bestSq) {
                bestSq = dsq;
                bestRoad = r.id;
                bestPt = i;
            }
        }
    }

    if (bestRoad == -1) return false;
    outRoadId = bestRoad;
    outPointIndex = bestPt;
    return true;
}

static bool SnapToAnyEndpoint(
    const std::vector<Road>& roads,
    const glm::vec3& p,
    float radius,
    glm::vec3& outSnap,
    int& outRoadId,
    bool& outIsStart)
{
    float bestSq = radius * radius;
    int bestRoad = -1;
    bool bestStart = false;
    glm::vec3 bestPos = p;

    for (const auto& r : roads) {
        if (r.pts.size() < 2) continue;
        glm::vec3 a = r.pts.front();
        glm::vec3 b = r.pts.back();

        glm::vec2 da(p.x - a.x, p.z - a.z);
        float dsa = da.x*da.x + da.y*da.y;
        if (dsa < bestSq) { bestSq = dsa; bestRoad = r.id; bestStart = true; bestPos = a; }

        glm::vec2 db(p.x - b.x, p.z - b.z);
        float dsb = db.x*db.x + db.y*db.y;
        if (dsb < bestSq) { bestSq = dsb; bestRoad = r.id; bestStart = false; bestPos = b; }
    }

    if (bestRoad == -1) return false;
    outSnap = bestPos;
    outRoadId = bestRoad;
    outIsStart = bestStart;
    return true;
}

struct BuildingInstance {
    AssetId asset = 0;
    glm::vec3 localPos{};
    float yaw = 0.0f;
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    uint32_t seed = 0;
};

struct HouseAnim {
    glm::vec3 pos;
    float spawnTime;
    glm::vec3 forward;
    AssetId asset = 0;
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    uint32_t seed = 0;
};

struct ZoneChunk {
    static constexpr int DIM = 128;
    std::array<uint8_t, DIM * DIM> cells{};
    void clear() { cells.fill(0); }
    void set(int x, int z, uint8_t v) {
        if (x < 0 || x >= DIM || z < 0 || z >= DIM) return;
        cells[z * DIM + x] = v;
    }
    uint8_t get(int x, int z) const {
        if (x < 0 || x >= DIM || z < 0 || z >= DIM) return 0;
        return cells[z * DIM + x];
    }
};

struct WaterChunk {
    static constexpr int DIM = ZoneChunk::DIM;
    std::array<uint8_t, DIM * DIM> cells{};
    void clear() { cells.fill(0); }
    void set(int x, int z, uint8_t v) {
        if (x < 0 || x >= DIM || z < 0 || z >= DIM) return;
        cells[z * DIM + x] = v;
    }
    uint8_t get(int x, int z) const {
        if (x < 0 || x >= DIM || z < 0 || z >= DIM) return 0;
        return cells[z * DIM + x];
    }
};

constexpr uint8_t ZONE_FLAG_BUILDABLE = 1 << 0;
constexpr uint8_t ZONE_FLAG_ZONED = 1 << 1;
constexpr uint8_t ZONE_FLAG_BLOCKED = 1 << 2;
constexpr float ZONE_CELL_M = CHUNK_SIZE_M / ZoneChunk::DIM;
constexpr int   ZONE_DEPTH_CELLS = 6;
constexpr float ZONE_DEPTH_M = ZONE_DEPTH_CELLS * ZONE_CELL_M; // 48m with 8m cells
constexpr float ROAD_WIDTH_M = 16.0f;
constexpr float ROAD_HALF_M = ROAD_WIDTH_M * 0.5f;
constexpr float INTERSECTION_CLEAR_M = ROAD_HALF_M + ZONE_CELL_M * 0.5f;
constexpr float ROAD_TEX_TILE_M = ROAD_WIDTH_M;
constexpr float WATER_SURFACE_Y = 0.02f;
constexpr uint8_t ZONE_TYPE_SHIFT = 3;
constexpr uint8_t ZONE_TYPE_MASK = 0x18; // 2 bits for 4 zone types

enum class ZoneType : uint8_t {
    Residential = 0,
    Commercial = 1,
    Industrial = 2,
    Office = 3
};

static uint8_t ZoneTypeBits(ZoneType t) {
    return (uint8_t(t) << ZONE_TYPE_SHIFT) & ZONE_TYPE_MASK;
}

static ZoneType ZoneTypeFromFlags(uint8_t flags) {
    return (ZoneType)((flags & ZONE_TYPE_MASK) >> ZONE_TYPE_SHIFT);
}

static const char* ZoneTypeName(ZoneType t) {
    switch (t) {
        case ZoneType::Commercial: return "Commercial";
        case ZoneType::Industrial: return "Industrial";
        case ZoneType::Office: return "Office";
        default: return "Residential";
    }
}

static const char* ZoneTypeCategory(ZoneType t) {
    switch (t) {
        case ZoneType::Commercial: return "commercial";
        case ZoneType::Industrial: return "industrial";
        case ZoneType::Office: return "office";
        default: return "residential";
    }
}

static glm::vec3 BaseSizeForZone(ZoneType t) {
    switch (t) {
        case ZoneType::Commercial: return glm::vec3(12.0f, 8.0f, 14.0f);
        case ZoneType::Industrial: return glm::vec3(14.0f, 8.0f, 20.0f);
        case ZoneType::Office: return glm::vec3(25.0f, 30.0f, 25.0f); // ~10 stories
        default: return glm::vec3(8.0f, 6.0f, 12.0f);
    }
}

struct ZoneStrip {
    int id = 0;
    int roadId = 0;
    float d0 = 0.0f;
    float d1 = 0.0f;
    int sideMask = 3; // 1 = left, 2 = right, 3 = both
    ZoneType type = ZoneType::Residential;
    float depth = ZONE_DEPTH_M;
};

struct LotCell {
    int roadId = -1;
    int side = 0;          // -1 left, +1 right
    float d0 = 0.0f;       // start along road
    float d1 = 0.0f;       // end along road
    glm::vec3 center{};
    glm::vec3 forward{};
    glm::vec3 right{};
    bool zoned = false;
    ZoneType zoneType = ZoneType::Residential;
};

struct BuildingChunk {
    std::unordered_map<AssetId, std::vector<BuildingInstance>> instancesByAsset;
};

struct MinimapState {
    GLuint texture = 0;
    int size = 512;
    bool dirty = true;
};

struct AppState {
    int nextRoadId = 1;
    int nextZoneId = 1;

    std::vector<Road> roads;
    std::vector<ZoneStrip> zones;
    std::vector<LotCell> lots;
    std::unordered_map<uint64_t, std::vector<int>> lotIndicesByChunk;
    std::unordered_map<uint64_t, std::vector<glm::mat4>> houseStaticByChunk;
    std::unordered_map<uint64_t, BuildingChunk> buildingChunks;
    std::unordered_set<uint64_t> dirtyBuildingChunks;
    std::unordered_map<uint64_t, ZoneChunk> zoneChunks;
    std::unordered_set<uint64_t> dirtyZoneChunks;
    std::unordered_map<uint64_t, WaterChunk> waterChunks;
    std::unordered_map<uint64_t, std::vector<glm::vec3>> overlayBuildableByChunk;
    std::unordered_map<uint64_t, std::vector<glm::vec3>> overlayZonedResByChunk;
    std::unordered_map<uint64_t, std::vector<glm::vec3>> overlayZonedComByChunk;
    std::unordered_map<uint64_t, std::vector<glm::vec3>> overlayZonedIndByChunk;
    std::unordered_map<uint64_t, std::vector<glm::vec3>> overlayZonedOfficeByChunk;

    bool roadsDirty = true;
    bool zonesDirty = true;
    bool housesDirty = true;
    bool overlayDirty = true;

    std::vector<RoadVertex> roadMeshVerts;
    std::vector<glm::vec3> zonePreviewVerts;

    std::vector<glm::mat4> houseStatic;
    std::vector<HouseAnim> houseAnim;
};

static bool ZonesOverlap(float a0, float a1, float b0, float b1) {
    float lo = std::max(std::min(a0, a1), std::min(b0, b1));
    float hi = std::min(std::max(a0, a1), std::max(b0, b1));
    return hi >= lo;
}

static bool IsLotZoned(const AppState& s, const LotCell& lot, ZoneType& outType) {
    int sideBit = (lot.side < 0) ? 1 : 2;
    for (const auto& z : s.zones) {
        if (z.roadId != lot.roadId) continue;
        if (!(z.sideMask & sideBit)) continue;
        if (!ZonesOverlap(lot.d0, lot.d1, z.d0, z.d1)) continue;
        outType = z.type;
        return true;
    }
    return false;
}

static bool ZoneOverlapsExisting(const AppState& s, int roadId, float d0, float d1) {
    for (const auto& z : s.zones) {
        if (z.roadId != roadId) continue;
        if (ZonesOverlap(d0, d1, z.d0, z.d1)) return true;
    }
    return false;
}

static bool WorldToZoneCell(const glm::vec3& p, int& outCx, int& outCz, int& outXi, int& outZi);

static uint8_t GetZoneFlagsAt(const AppState& s, const glm::vec3& pos) {
    ChunkCoord cc = ChunkFromPosXZ(pos);
    uint64_t key = PackChunk(cc.cx, cc.cz);
    auto it = s.zoneChunks.find(key);
    if (it == s.zoneChunks.end()) return 0;
    float originX = cc.cx * CHUNK_SIZE_M;
    float originZ = cc.cz * CHUNK_SIZE_M;
    int xi = (int)std::floor((pos.x - originX) / ZONE_CELL_M);
    int zi = (int)std::floor((pos.z - originZ) / ZONE_CELL_M);
    return it->second.get(xi, zi);
}

static uint8_t GetWaterAt(const AppState& s, const glm::vec3& pos) {
    int cx, cz, xi, zi;
    if (!WorldToZoneCell(pos, cx, cz, xi, zi)) return 0;
    uint64_t key = PackChunk(cx, cz);
    auto it = s.waterChunks.find(key);
    if (it == s.waterChunks.end()) return 0;
    return it->second.get(xi, zi);
}

static ZoneChunk& EnsureZoneChunk(AppState& s, uint64_t key);
static WaterChunk& EnsureWaterChunk(AppState& s, uint64_t key);

static bool WorldToZoneCell(const glm::vec3& p, int& outCx, int& outCz, int& outXi, int& outZi) {
    int cx = (int)std::floor(p.x / CHUNK_SIZE_M);
    int cz = (int)std::floor(p.z / CHUNK_SIZE_M);

    float originX = cx * CHUNK_SIZE_M;
    float originZ = cz * CHUNK_SIZE_M;

    int xi = (int)std::floor((p.x - originX) / ZONE_CELL_M);
    int zi = (int)std::floor((p.z - originZ) / ZONE_CELL_M);

    if (xi < 0 || xi >= ZoneChunk::DIM || zi < 0 || zi >= ZoneChunk::DIM) return false;

    outCx = cx;
    outCz = cz;
    outXi = xi;
    outZi = zi;
    return true;
}

static void SetZoneCellFlags(
    AppState& s,
    int cx,
    int cz,
    int xi,
    int zi,
    uint8_t setMask,
    uint8_t clearMask)
{
    uint64_t key = PackChunk(cx, cz);
    ZoneChunk& chunk = EnsureZoneChunk(s, key);
    uint8_t v = chunk.get(xi, zi);
    v &= (uint8_t)~clearMask;
    v |= setMask;
    chunk.set(xi, zi, v);
    s.dirtyZoneChunks.insert(key);
}

static float ZoneRectCoverage(
    const AppState& s,
    const glm::vec3& center,
    const glm::vec3& forward,
    const glm::vec3& right,
    float width,
    float depth,
    uint8_t requiredMask,
    uint8_t forbiddenMask)
{
    int nx = std::max(1, (int)std::ceil(width / ZONE_CELL_M));
    int nz = std::max(1, (int)std::ceil(depth / ZONE_CELL_M));
    float stepX = width / (float)nx;
    float stepZ = depth / (float)nz;
    float halfW = width * 0.5f;
    float halfD = depth * 0.5f;
    int total = nx * nz;
    int hit = 0;

    for (int iz = 0; iz < nz; iz++) {
        float v = -halfD + (iz + 0.5f) * stepZ;
        for (int ix = 0; ix < nx; ix++) {
            float u = -halfW + (ix + 0.5f) * stepX;
            glm::vec3 p = center + right * u + forward * v;
            uint8_t flags = GetZoneFlagsAt(s, p);
            if (flags & forbiddenMask) return 0.0f;
            if ((flags & requiredMask) == requiredMask) hit++;
        }
    }
    return total > 0 ? (float)hit / (float)total : 0.0f;
}

[[maybe_unused]] static float ZoneRectTypeCoverage(
    const AppState& s,
    const glm::vec3& center,
    const glm::vec3& forward,
    const glm::vec3& right,
    float width,
    float depth,
    ZoneType type,
    uint8_t requiredMask,
    uint8_t forbiddenMask)
{
    int nx = std::max(1, (int)std::ceil(width / ZONE_CELL_M));
    int nz = std::max(1, (int)std::ceil(depth / ZONE_CELL_M));
    float stepX = width / (float)nx;
    float stepZ = depth / (float)nz;
    float halfW = width * 0.5f;
    float halfD = depth * 0.5f;
    int total = nx * nz;
    int hit = 0;

    for (int iz = 0; iz < nz; iz++) {
        float v = -halfD + (iz + 0.5f) * stepZ;
        for (int ix = 0; ix < nx; ix++) {
            float u = -halfW + (ix + 0.5f) * stepX;
            glm::vec3 p = center + right * u + forward * v;
            uint8_t flags = GetZoneFlagsAt(s, p);
            if (flags & forbiddenMask) return 0.0f;
            if ((flags & requiredMask) != requiredMask) continue;
            if (ZoneTypeFromFlags(flags) == type) hit++;
        }
    }
    return total > 0 ? (float)hit / (float)total : 0.0f;
}

[[maybe_unused]] static ZoneType ZoneRectMajorityType(
    const AppState& s,
    const glm::vec3& center,
    const glm::vec3& forward,
    const glm::vec3& right,
    float width,
    float depth)
{
    int nx = std::max(1, (int)std::ceil(width / ZONE_CELL_M));
    int nz = std::max(1, (int)std::ceil(depth / ZONE_CELL_M));
    float stepX = width / (float)nx;
    float stepZ = depth / (float)nz;
    float halfW = width * 0.5f;
    float halfD = depth * 0.5f;
    std::array<int, 4> counts{};

    for (int iz = 0; iz < nz; iz++) {
        float v = -halfD + (iz + 0.5f) * stepZ;
        for (int ix = 0; ix < nx; ix++) {
            float u = -halfW + (ix + 0.5f) * stepX;
            glm::vec3 p = center + right * u + forward * v;
            uint8_t flags = GetZoneFlagsAt(s, p);
            if (!(flags & ZONE_FLAG_ZONED)) continue;
            int idx = (int)ZoneTypeFromFlags(flags);
            if (idx >= 0 && idx < (int)counts.size()) counts[idx]++;
        }
    }

    int best = 0;
    int bestCount = -1;
    for (int i = 0; i < (int)counts.size(); i++) {
        if (counts[i] > bestCount) {
            bestCount = counts[i];
            best = i;
        }
    }
    return (ZoneType)best;
}

static bool LotRectMeetsGrid(
    const AppState& s,
    const glm::vec3& center,
    const glm::vec3& forward,
    const glm::vec3& right,
    float width,
    float depth,
    uint8_t requiredMask,
    uint8_t forbiddenMask,
    float minCoverage)
{
    return ZoneRectCoverage(s, center, forward, right, width, depth, requiredMask, forbiddenMask) >= minCoverage;
}

static ZoneChunk& EnsureZoneChunk(AppState& s, uint64_t key) {
    auto it = s.zoneChunks.find(key);
    if (it == s.zoneChunks.end()) {
        ZoneChunk z;
        z.clear();
        it = s.zoneChunks.emplace(key, std::move(z)).first;
    }
    return it->second;
}

static WaterChunk& EnsureWaterChunk(AppState& s, uint64_t key) {
    auto it = s.waterChunks.find(key);
    if (it == s.waterChunks.end()) {
        WaterChunk w;
        w.clear();
        it = s.waterChunks.emplace(key, std::move(w)).first;
    }
    return it->second;
}

[[maybe_unused]] static void StampZoneStrip(AppState& s, const ZoneStrip& z, bool add) {
    int ridx = FindRoadIndexById(s.roads, z.roadId);
    if (ridx < 0) return;
    const Road& r = s.roads[ridx];
    if (r.pts.size() < 2) return;
    float dA = std::min(z.d0, z.d1);
    float dB = std::max(z.d0, z.d1);
    const float stepAlong = ZONE_CELL_M * 0.5f;

    for (float d = dA; d <= dB; d += stepAlong) {
        glm::vec3 tan;
        glm::vec3 p = r.pointAt(d, tan);
        if (glm::dot(tan, tan) < 1e-6f) continue;

        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), tan));

        auto stampSide = [&](int side, int sideBit) {
            if (!(z.sideMask & sideBit)) return;
            for (int row = 0; row < ZONE_DEPTH_CELLS; ++row) {
                float offset = ROAD_HALF_M + (row + 0.5f) * ZONE_CELL_M;
                glm::vec3 sample = p + right * (float(side) * offset);

                int cx, cz, xi, zi;
                if (!WorldToZoneCell(sample, cx, cz, xi, zi)) continue;
                uint8_t flags = GetZoneFlagsAt(s, sample);
                if (!(flags & ZONE_FLAG_BUILDABLE)) continue;
                if (flags & ZONE_FLAG_BLOCKED) continue;

                uint8_t setMask = add ? (uint8_t)(ZONE_FLAG_ZONED | ZoneTypeBits(z.type)) : 0;
                uint8_t clrMask = add ? ZONE_TYPE_MASK : (uint8_t)(ZONE_FLAG_ZONED | ZONE_TYPE_MASK);
                SetZoneCellFlags(s, cx, cz, xi, zi, setMask, clrMask);
                s.dirtyBuildingChunks.insert(PackChunk(cx, cz));
            }
        };

        stampSide(-1, 1);
        stampSide(+1, 2);
    }
}

[[maybe_unused]] static void StampBlockedDisk(AppState& s, const glm::vec3& center, float radiusM) {
    float minX = center.x - radiusM;
    float maxX = center.x + radiusM;
    float minZ = center.z - radiusM;
    float maxZ = center.z + radiusM;

    ChunkCoord cmin = ChunkFromPosXZ(glm::vec3(minX, 0, minZ));
    ChunkCoord cmax = ChunkFromPosXZ(glm::vec3(maxX, 0, maxZ));

    float r2 = radiusM * radiusM;

    for (int cz = cmin.cz; cz <= cmax.cz; ++cz) {
        for (int cx = cmin.cx; cx <= cmax.cx; ++cx) {
            uint64_t key = PackChunk(cx, cz);
            ZoneChunk& chunk = EnsureZoneChunk(s, key);
            float originX = cx * CHUNK_SIZE_M;
            float originZ = cz * CHUNK_SIZE_M;

            int x0 = std::max(0, (int)std::floor((minX - originX) / ZONE_CELL_M));
            int x1 = std::min(ZoneChunk::DIM - 1, (int)std::floor((maxX - originX) / ZONE_CELL_M));
            int z0 = std::max(0, (int)std::floor((minZ - originZ) / ZONE_CELL_M));
            int z1 = std::min(ZoneChunk::DIM - 1, (int)std::floor((maxZ - originZ) / ZONE_CELL_M));

            for (int zi = z0; zi <= z1; ++zi) {
                for (int xi = x0; xi <= x1; ++xi) {
                    glm::vec3 cellCenter(
                        originX + (xi + 0.5f) * ZONE_CELL_M,
                        0.0f,
                        originZ + (zi + 0.5f) * ZONE_CELL_M
                    );

                    glm::vec2 d(cellCenter.x - center.x, cellCenter.z - center.z);
                    if (d.x*d.x + d.y*d.y > r2) continue;

                    uint8_t v = chunk.get(xi, zi);
                    v |= ZONE_FLAG_BLOCKED;
                    v &= (uint8_t)~ZONE_FLAG_BUILDABLE;
                    v &= (uint8_t)~ZONE_FLAG_ZONED;
                    v &= (uint8_t)~ZONE_TYPE_MASK;
                    chunk.set(xi, zi, v);
                    s.dirtyZoneChunks.insert(key);
                    s.dirtyBuildingChunks.insert(key);
                }
            }
        }
    }
}

static void StampRoadInfluence(AppState& s, const Road& r) {
    if (r.pts.size() < 2) return;

    float total = r.totalLen();
    const float stepAlong = ZONE_CELL_M * 0.5f;

    for (float d = 0.0f; d <= total; d += stepAlong) {
        glm::vec3 tan;
        glm::vec3 p = r.pointAt(d, tan);
        if (glm::dot(tan, tan) < 1e-6f) continue;

        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), tan));

        for (int side : {-1, +1}) {
            for (int row = 0; row < ZONE_DEPTH_CELLS; ++row) {
                float offset = ROAD_HALF_M + (row + 0.5f) * ZONE_CELL_M;
                glm::vec3 sample = p + right * (float(side) * offset);

                int cx, cz, xi, zi;
                if (!WorldToZoneCell(sample, cx, cz, xi, zi)) continue;
                SetZoneCellFlags(s, cx, cz, xi, zi, ZONE_FLAG_BUILDABLE, 0);
            }
        }
    }
}

static void StampRoadSurfaceBlocked(AppState& s, const Road& r) {
    if (r.pts.size() < 2) return;

    float total = r.totalLen();
    const float stepAlong = ZONE_CELL_M * 0.5f;
    const float stepAcross = ZONE_CELL_M * 0.5f;

    for (float d = 0.0f; d <= total; d += stepAlong) {
        glm::vec3 tan;
        glm::vec3 p = r.pointAt(d, tan);
        if (glm::dot(tan, tan) < 1e-6f) continue;

        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), tan));
        for (float off = -ROAD_HALF_M; off <= ROAD_HALF_M; off += stepAcross) {
            glm::vec3 sample = p + right * off;

            int cx, cz, xi, zi;
            if (!WorldToZoneCell(sample, cx, cz, xi, zi)) continue;
            SetZoneCellFlags(
                s, cx, cz, xi, zi,
                ZONE_FLAG_BLOCKED,
                (uint8_t)(ZONE_FLAG_BUILDABLE | ZONE_FLAG_ZONED | ZONE_TYPE_MASK));
            s.dirtyBuildingChunks.insert(PackChunk(cx, cz));
        }
    }
}

static void StampWaterMask(AppState& s) {
    if (s.waterChunks.empty()) return;
    for (const auto& kv : s.waterChunks) {
        int32_t cx, cz;
        UnpackChunk(kv.first, cx, cz);
        const WaterChunk& chunk = kv.second;
        for (int zi = 0; zi < WaterChunk::DIM; ++zi) {
            for (int xi = 0; xi < WaterChunk::DIM; ++xi) {
                if (chunk.get(xi, zi) == 0) continue;
                SetZoneCellFlags(
                    s, cx, cz, xi, zi,
                    ZONE_FLAG_BLOCKED,
                    (uint8_t)(ZONE_FLAG_BUILDABLE | ZONE_FLAG_ZONED | ZONE_TYPE_MASK));
                s.dirtyBuildingChunks.insert(PackChunk(cx, cz));
            }
        }
    }
}

static bool LoadWaterMaskFromImage(AppState& s, const char* path, float threshold) {
    std::vector<uint8_t> pixels;
    int w = 0;
    int h = 0;
    if (!LoadImageRGBA(path, pixels, w, h)) return false;
    if (w <= 0 || h <= 0) return false;

    s.waterChunks.clear();

    const float mapHalf = MAP_HALF_M;
    const float invMap = 1.0f / MAP_SIDE_M;
    const float startX = -mapHalf + ZONE_CELL_M * 0.5f;
    const float startZ = -mapHalf + ZONE_CELL_M * 0.5f;
    const int cellsPerSide = (int)std::ceil(MAP_SIDE_M / ZONE_CELL_M);

    int waterCells = 0;
    for (int gz = 0; gz < cellsPerSide; ++gz) {
        float wz = startZ + gz * ZONE_CELL_M;
        float v = 1.0f - ((wz + mapHalf) * invMap);
        if (v < 0.0f || v > 1.0f) continue;
        int pz = (int)Clamp(std::floor(v * (float)h), 0.0f, (float)(h - 1));
        for (int gx = 0; gx < cellsPerSide; ++gx) {
            float wx = startX + gx * ZONE_CELL_M;
            float u = (wx + mapHalf) * invMap;
            if (u < 0.0f || u > 1.0f) continue;
            int px = (int)Clamp(std::floor(u * (float)w), 0.0f, (float)(w - 1));
            const uint8_t* p = &pixels[(pz * w + px) * 4];
            float lum = (p[0] + p[1] + p[2]) * (1.0f / (3.0f * 255.0f));
            if (lum < threshold) continue;

            glm::vec3 sample(wx, 0.0f, wz);
            int cx, cz, xi, zi;
            if (!WorldToZoneCell(sample, cx, cz, xi, zi)) continue;
            WaterChunk& wc = EnsureWaterChunk(s, PackChunk(cx, cz));
            if (wc.get(xi, zi) == 0) {
                wc.set(xi, zi, 1);
                waterCells++;
            }
        }
    }

    SDL_Log("Water mask loaded: %d cells from %s", waterCells, path);
    s.zonesDirty = true;
    s.housesDirty = true;
    s.overlayDirty = true;
    return true;
}

static void UpdateMinimapTexture(MinimapState& mm, const AppState& s) {
    if (!mm.dirty && mm.texture != 0) return;
    if (mm.size <= 0) return;

    if (mm.texture == 0) {
        glGenTextures(1, &mm.texture);
    }

    std::vector<uint8_t> pixels;
    pixels.resize((size_t)mm.size * (size_t)mm.size * 4);
    const float mapHalf = MAP_HALF_M;
    const float invMap = 1.0f / MAP_SIDE_M;
    const uint8_t land[3] = { 32, 96, 40 };
    const uint8_t water[3] = { 40, 80, 120 };

    const bool hasWater = !s.waterChunks.empty();
    for (int y = 0; y < mm.size; ++y) {
        float v = (y + 0.5f) / (float)mm.size;
        float wz = (0.5f - v) * MAP_SIDE_M;
        for (int x = 0; x < mm.size; ++x) {
            float u = (x + 0.5f) / (float)mm.size;
            float wx = (u - 0.5f) * MAP_SIDE_M;
            bool isWater = false;
            if (hasWater) {
                isWater = GetWaterAt(s, glm::vec3(wx, 0.0f, wz)) != 0;
            }
            const uint8_t* c = isWater ? water : land;
            size_t idx = (size_t)(y * mm.size + x) * 4;
            pixels[idx + 0] = c[0];
            pixels[idx + 1] = c[1];
            pixels[idx + 2] = c[2];
            pixels[idx + 3] = 255;
        }
    }

    glBindTexture(GL_TEXTURE_2D, mm.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mm.size, mm.size, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    mm.dirty = false;
}

static void RebuildZoneGrid(AppState& s) {
    s.zoneChunks.clear();
    s.dirtyZoneChunks.clear();
    if (s.roads.empty()) return;

    for (const auto& r : s.roads) {
        StampRoadInfluence(s, r);
        StampRoadSurfaceBlocked(s, r);
    }
    StampWaterMask(s);
}

// --- Undo/Redo command system ---
struct ICommand {
    virtual ~ICommand() = default;
    virtual const char* name() const = 0;
    virtual void doIt(AppState& s) = 0;
    virtual void undoIt(AppState& s) = 0;
};

struct CmdAddRoad : ICommand {
    Road road;
    bool applied = false;

    CmdAddRoad(const Road& r) : road(r) {}
    const char* name() const override { return "AddRoad"; }

    void doIt(AppState& s) override {
        if (!applied) {
            s.roads.push_back(road);
            applied = true;
        }
        s.roadsDirty = true;
    }

    void undoIt(AppState& s) override {
        int idx = FindRoadIndexById(s.roads, road.id);
        if (idx >= 0) s.roads.erase(s.roads.begin() + idx);
        s.roadsDirty = true;
        s.housesDirty = true; // zones might refer to a removed road; for simplicity, we keep zones but houses will rebuild and skip invalid
    }
};

struct CmdExtendRoad : ICommand {
    int roadId = 0;
    std::vector<glm::vec3> added;
    bool atStart = false;

    CmdExtendRoad(int rid, const std::vector<glm::vec3>& pts, bool start)
        : roadId(rid), added(pts), atStart(start) {}

    const char* name() const override { return "ExtendRoad"; }

    void doIt(AppState& s) override {
        int idx = FindRoadIndexById(s.roads, roadId);
        if (idx < 0) return;
        Road& r = s.roads[idx];

        if (added.empty()) return;
        if (atStart) {
            for (int i = (int)added.size() - 1; i >= 0; i--) {
                r.pts.insert(r.pts.begin(), added[i]);
            }
        } else {
            for (auto& p : added) r.pts.push_back(p);
        }
        r.rebuildCum();
        s.roadsDirty = true;
    }

    void undoIt(AppState& s) override {
        int idx = FindRoadIndexById(s.roads, roadId);
        if (idx < 0) return;
        Road& r = s.roads[idx];

        if ((int)r.pts.size() <= (int)added.size()) return;
        if (atStart) {
            r.pts.erase(r.pts.begin(), r.pts.begin() + (int)added.size());
        } else {
            r.pts.erase(r.pts.end() - (int)added.size(), r.pts.end());
        }
        r.rebuildCum();
        s.roadsDirty = true;
    }
};

struct CmdMoveRoadPoint : ICommand {
    int roadId = 0;
    int pointIndex = -1;
    glm::vec3 oldPos{};
    glm::vec3 newPos{};

    CmdMoveRoadPoint(int rid, int pi, glm::vec3 a, glm::vec3 b)
        : roadId(rid), pointIndex(pi), oldPos(a), newPos(b) {}

    const char* name() const override { return "MoveRoadPoint"; }

    void doIt(AppState& s) override {
        int idx = FindRoadIndexById(s.roads, roadId);
        if (idx < 0) return;
        Road& r = s.roads[idx];
        if (pointIndex < 0 || pointIndex >= (int)r.pts.size()) return;
        r.pts[pointIndex] = newPos;
        r.pts[pointIndex].y = 0.0f;
        r.rebuildCum();
        s.roadsDirty = true;
        s.housesDirty = true;
    }

    void undoIt(AppState& s) override {
        int idx = FindRoadIndexById(s.roads, roadId);
        if (idx < 0) return;
        Road& r = s.roads[idx];
        if (pointIndex < 0 || pointIndex >= (int)r.pts.size()) return;
        r.pts[pointIndex] = oldPos;
        r.pts[pointIndex].y = 0.0f;
        r.rebuildCum();
        s.roadsDirty = true;
        s.housesDirty = true;
    }
};

struct CmdDeleteRoadPoint : ICommand {
    int roadId = 0;
    int pointIndex = -1;
    glm::vec3 removed{};
    bool did = false;

    CmdDeleteRoadPoint(int rid, int pi) : roadId(rid), pointIndex(pi) {}

    const char* name() const override { return "DeleteRoadPoint"; }

    void doIt(AppState& s) override {
        int idx = FindRoadIndexById(s.roads, roadId);
        if (idx < 0) return;
        Road& r = s.roads[idx];
        if (pointIndex < 0 || pointIndex >= (int)r.pts.size()) return;
        if ((int)r.pts.size() <= 2) return; // keep roads valid
        removed = r.pts[pointIndex];
        r.pts.erase(r.pts.begin() + pointIndex);
        r.rebuildCum();
        did = true;
        s.roadsDirty = true;
        s.housesDirty = true;
    }

    void undoIt(AppState& s) override {
        if (!did) return;
        int idx = FindRoadIndexById(s.roads, roadId);
        if (idx < 0) return;
        Road& r = s.roads[idx];
        pointIndex = Clamp((float)pointIndex, 0.0f, (float)r.pts.size());
        r.pts.insert(r.pts.begin() + pointIndex, removed);
        r.rebuildCum();
        s.roadsDirty = true;
        s.housesDirty = true;
    }
};

struct CmdAddZone : ICommand {
    ZoneStrip zone;
    bool applied = false;

    CmdAddZone(const ZoneStrip& z) : zone(z) {}
    const char* name() const override { return "AddZone"; }

    void doIt(AppState& s) override {
        if (!applied) {
            s.zones.push_back(zone);
            applied = true;
        }
        s.zonesDirty = true;
        s.housesDirty = true;
    }

    void undoIt(AppState& s) override {
        auto it = std::find_if(s.zones.begin(), s.zones.end(), [&](const ZoneStrip& z){ return z.id == zone.id; });
        if (it != s.zones.end()) s.zones.erase(it);
        s.zonesDirty = true;
        s.housesDirty = true;
    }
};

struct CmdClearZonesForRoad : ICommand {
    int roadId = -1;
    std::vector<ZoneStrip> removed;
    bool applied = false;

    CmdClearZonesForRoad(int rid, const std::vector<ZoneStrip>& zs) : roadId(rid), removed(zs) {}
    const char* name() const override { return "ClearZones"; }

    void doIt(AppState& s) override {
        if (!applied) {
            s.zones.erase(std::remove_if(s.zones.begin(), s.zones.end(),
                                         [&](const ZoneStrip& z){ return z.roadId == roadId; }),
                          s.zones.end());
            applied = true;
        }
        s.zonesDirty = true;
        s.housesDirty = true;
    }

    void undoIt(AppState& s) override {
        for (const auto& z : removed) s.zones.push_back(z);
        s.zonesDirty = true;
        s.housesDirty = true;
    }
};

struct CommandStack {
    std::vector<std::unique_ptr<ICommand>> undo;
    std::vector<std::unique_ptr<ICommand>> redo;

    void exec(AppState& s, std::unique_ptr<ICommand> cmd) {
        cmd->doIt(s);
        undo.push_back(std::move(cmd));
        redo.clear();
    }

    void doUndo(AppState& s) {
        if (undo.empty()) return;
        auto cmd = std::move(undo.back());
        undo.pop_back();
        cmd->undoIt(s);
        redo.push_back(std::move(cmd));
    }

    void doRedo(AppState& s) {
        if (redo.empty()) return;
        auto cmd = std::move(redo.back());
        redo.pop_back();
        cmd->doIt(s);
        undo.push_back(std::move(cmd));
    }

    void clear() {
        undo.clear();
        redo.clear();
    }
};

static void RebuildAllRoadMesh(AppState& s) {
    s.roadMeshVerts.clear();
    const float roadWidth = ROAD_WIDTH_M;
    const float y = 0.03f;

    for (const auto& r : s.roads) {
        if (r.pts.size() < 2) continue;
        float vAccum = 0.0f;
        for (size_t i = 0; i + 1 < r.pts.size(); i++) {
            glm::vec3 a = r.pts[i];
            glm::vec3 b = r.pts[i+1];

            glm::vec3 dir = b - a;
            dir.y = 0.0f;
            float l = std::sqrt(dir.x*dir.x + dir.z*dir.z);
            if (l < 1e-4f) continue;
            dir /= l;

            glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), dir));
            glm::vec3 off = right * (roadWidth * 0.5f);

            glm::vec3 aL = a - off; aL.y = y;
            glm::vec3 aR = a + off; aR.y = y;
            glm::vec3 bL = b - off; bL.y = y;
            glm::vec3 bR = b + off; bR.y = y;

            float v0 = vAccum / ROAD_TEX_TILE_M;
            float v1 = (vAccum + l) / ROAD_TEX_TILE_M;
            vAccum += l;

            s.roadMeshVerts.push_back({aL, glm::vec2(0.0f, v0)});
            s.roadMeshVerts.push_back({aR, glm::vec2(1.0f, v0)});
            s.roadMeshVerts.push_back({bR, glm::vec2(1.0f, v1)});

            s.roadMeshVerts.push_back({aL, glm::vec2(0.0f, v0)});
            s.roadMeshVerts.push_back({bR, glm::vec2(1.0f, v1)});
            s.roadMeshVerts.push_back({bL, glm::vec2(0.0f, v1)});
        }
    }
}

[[maybe_unused]] static void AppendZoneMesh(
    std::vector<glm::vec3>& out,
    const Road& r,
    float d0,
    float d1,
    int sideMask,
    float depth)
{
    float a = std::min(d0, d1);
    float b = std::max(d0, d1);
    if (b - a < 1.0f) return;

    const float roadHalf = ROAD_HALF_M;
    const float setback = roadHalf + 1.0f;
    const float step = 6.0f;
    const float y = 0.04f;

    auto emitStrip = [&](int side) {
        for (float d = a; d <= b - step; d += step) {
            glm::vec3 t0, t1;
            glm::vec3 p0 = r.pointAt(d, t0);
            glm::vec3 p1 = r.pointAt(d + step, t1);

            glm::vec3 right0 = glm::normalize(glm::cross(glm::vec3(0,1,0), t0));
            glm::vec3 right1 = glm::normalize(glm::cross(glm::vec3(0,1,0), t1));

            glm::vec3 in0  = p0 + right0 * float(side) * setback;
            glm::vec3 out0 = p0 + right0 * float(side) * (setback + depth);
            glm::vec3 in1  = p1 + right1 * float(side) * setback;
            glm::vec3 out1 = p1 + right1 * float(side) * (setback + depth);

            in0.y = y; out0.y = y; in1.y = y; out1.y = y;

            out.push_back(in0);
            out.push_back(out0);
            out.push_back(out1);

            out.push_back(in0);
            out.push_back(out1);
            out.push_back(in1);
        }
    };

    if (sideMask & 1) emitStrip(-1);
    if (sideMask & 2) emitStrip(+1);
}

[[maybe_unused]] static void AppendLotOverlayQuad(std::vector<glm::vec3>& out, const glm::vec3& center, const glm::vec3& forward, const glm::vec3& right, float width, float depth) {
    const float y = 0.04f;
    glm::vec3 f = forward;
    if (glm::dot(f, f) < 1e-6f) f = glm::vec3(0,0,1);
    glm::vec3 r = right;
    if (glm::dot(r, r) < 1e-6f) r = glm::vec3(1,0,0);
    glm::vec3 fOff = glm::normalize(f) * (width * 0.5f);
    glm::vec3 rOff = glm::normalize(r) * (depth * 0.5f);

    glm::vec3 a = center - fOff - rOff; a.y = y;
    glm::vec3 b = center + fOff - rOff; b.y = y;
    glm::vec3 c = center + fOff + rOff; c.y = y;
    glm::vec3 d = center - fOff + rOff; d.y = y;

    out.push_back(a); out.push_back(b); out.push_back(c);
    out.push_back(a); out.push_back(c); out.push_back(d);
}

[[maybe_unused]] static void AppendZoneCellQuad(std::vector<glm::vec3>& out, float originX, float originZ, int xi, int zi, float inset = 0.15f) {
    const float y = 0.04f;

    float x0 = originX + xi * ZONE_CELL_M + inset;
    float z0 = originZ + zi * ZONE_CELL_M + inset;
    float x1 = originX + (xi + 1) * ZONE_CELL_M - inset;
    float z1 = originZ + (zi + 1) * ZONE_CELL_M - inset;

    out.push_back({x0, y, z0}); out.push_back({x1, y, z0}); out.push_back({x1, y, z1});
    out.push_back({x0, y, z0}); out.push_back({x1, y, z1}); out.push_back({x0, y, z1});
}

static void AppendOrientedZoneCellQuad(
    std::vector<glm::vec3>& out,
    const glm::vec3& center,
    const glm::vec3& forward,
    const glm::vec3& away,
    float y = 0.04f,
    float inset = 0.15f)
{
    glm::vec3 f = forward;
    if (glm::dot(f, f) < 1e-6f) f = glm::vec3(1, 0, 0);
    glm::vec3 a = away;
    if (glm::dot(a, a) < 1e-6f) a = glm::vec3(0, 0, 1);
    f = glm::normalize(f);
    a = glm::normalize(a);

    float half = std::max(0.0f, ZONE_CELL_M * 0.5f - inset);
    glm::vec3 fOff = f * half;
    glm::vec3 aOff = a * half;

    glm::vec3 p0 = center - fOff - aOff; p0.y = y;
    glm::vec3 p1 = center + fOff - aOff; p1.y = y;
    glm::vec3 p2 = center + fOff + aOff; p2.y = y;
    glm::vec3 p3 = center - fOff + aOff; p3.y = y;

    out.push_back(p0); out.push_back(p1); out.push_back(p2);
    out.push_back(p0); out.push_back(p2); out.push_back(p3);
}

static void AppendWaterCellQuad(std::vector<glm::vec3>& out, float originX, float originZ, int xi, int zi, float inset = 0.02f) {
    const float y = WATER_SURFACE_Y;

    float x0 = originX + xi * ZONE_CELL_M + inset;
    float z0 = originZ + zi * ZONE_CELL_M + inset;
    float x1 = originX + (xi + 1) * ZONE_CELL_M - inset;
    float z1 = originZ + (zi + 1) * ZONE_CELL_M - inset;

    out.push_back({x0, y, z0}); out.push_back({x1, y, z0}); out.push_back({x1, y, z1});
    out.push_back({x0, y, z0}); out.push_back({x1, y, z1}); out.push_back({x0, y, z1});
}

static const ZoneStrip* FindZoneForRoadAt(const std::vector<const ZoneStrip*>& zones, float d, int sideBit) {
    for (const ZoneStrip* z : zones) {
        if (!(z->sideMask & sideBit)) continue;
        float lo = std::min(z->d0, z->d1);
        float hi = std::max(z->d0, z->d1);
        if (d >= lo && d <= hi) return z;
    }
    return nullptr;
}

static bool ShouldCullForIntersection(
    const AppState& s,
    int roadId,
    const glm::vec3& pos,
    const glm::vec3& forward,
    float clearDist)
{
    float fLenSq = glm::dot(forward, forward);
    if (fLenSq < 1e-6f) return false;
    glm::vec3 f = forward / std::sqrt(fLenSq);
    float clearSq = clearDist * clearDist;
    for (const auto& other : s.roads) {
        if (other.id == roadId) continue;
        if (other.pts.size() < 2) continue;
        float dAlong;
        glm::vec3 tan;
        float distSq = ClosestDistanceAlongRoadSq(other, pos, dAlong, tan);
        if (distSq >= clearSq) continue;
        float tLenSq = glm::dot(tan, tan);
        if (tLenSq < 1e-6f) return true;
        glm::vec3 t = tan / std::sqrt(tLenSq);
        float align = std::fabs(glm::dot(f, t));
        if (align > 0.85f) continue;
        return true;
    }
    return false;
}

static void RebuildRoadAlignedOverlay(AppState& s) {
    s.overlayBuildableByChunk.clear();
    s.overlayZonedResByChunk.clear();
    s.overlayZonedComByChunk.clear();
    s.overlayZonedIndByChunk.clear();
    s.overlayZonedOfficeByChunk.clear();

    if (s.roads.empty()) return;

    std::unordered_map<int, std::vector<const ZoneStrip*>> zonesByRoad;
    zonesByRoad.reserve(s.zones.size());
    for (const auto& z : s.zones) {
        zonesByRoad[z.roadId].push_back(&z);
    }

    for (const auto& r : s.roads) {
        if (r.pts.size() < 2) continue;
        float total = r.totalLen();
        int cols = (int)std::floor(total / ZONE_CELL_M);
        if (cols <= 0) continue;

        const std::vector<const ZoneStrip*>* zones = nullptr;
        auto zIt = zonesByRoad.find(r.id);
        if (zIt != zonesByRoad.end()) zones = &zIt->second;

        for (int i = 0; i < cols; ++i) {
            float d = (i + 0.5f) * ZONE_CELL_M;
            glm::vec3 tan;
            glm::vec3 pos = r.pointAt(d, tan);
            if (glm::dot(tan, tan) < 1e-6f) continue;

            glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0, 1, 0), tan));
            for (int side : {-1, +1}) {
                glm::vec3 away = right * (float)side;
                int sideBit = (side < 0) ? 1 : 2;
                const ZoneStrip* z = zones ? FindZoneForRoadAt(*zones, d, sideBit) : nullptr;

                for (int row = 0; row < ZONE_DEPTH_CELLS; ++row) {
                    float off = ROAD_HALF_M + (row + 0.5f) * ZONE_CELL_M;
                    glm::vec3 center = pos + away * off;
                    if (GetWaterAt(s, center) != 0) continue;
                    if (ShouldCullForIntersection(s, r.id, center, tan, INTERSECTION_CLEAR_M)) continue;

                    ChunkCoord cc = ChunkFromPosXZ(center);
                    uint64_t key = PackChunk(cc.cx, cc.cz);
                    AppendOrientedZoneCellQuad(s.overlayBuildableByChunk[key], center, tan, away);

                    if (!z) continue;
                    switch (z->type) {
                        case ZoneType::Commercial:
                            AppendOrientedZoneCellQuad(s.overlayZonedComByChunk[key], center, tan, away);
                            break;
                        case ZoneType::Industrial:
                            AppendOrientedZoneCellQuad(s.overlayZonedIndByChunk[key], center, tan, away);
                            break;
                        case ZoneType::Office:
                            AppendOrientedZoneCellQuad(s.overlayZonedOfficeByChunk[key], center, tan, away);
                            break;
                        default:
                            AppendOrientedZoneCellQuad(s.overlayZonedResByChunk[key], center, tan, away);
                            break;
                    }
                }
            }
        }
    }
}

struct PreviewCellKey {
    int32_t cx = 0;
    int32_t cz = 0;
    uint8_t xi = 0;
    uint8_t zi = 0;

    bool operator==(const PreviewCellKey& other) const {
        return cx == other.cx && cz == other.cz && xi == other.xi && zi == other.zi;
    }
};

struct PreviewCellKeyHash {
    std::size_t operator()(const PreviewCellKey& k) const {
        uint32_t h1 = Hash32((uint32_t)k.cx);
        uint32_t h2 = Hash32((uint32_t)k.cz);
        uint32_t h3 = Hash32((uint32_t(k.xi) << 16) | uint32_t(k.zi));
        uint32_t h = h1 ^ (h2 * 0x9e3779b1U) ^ (h3 * 0x85ebca6bU);
        return (std::size_t)h;
    }
};

static void BuildZonePreviewMesh(
    AppState& s,
    const Road& r,
    float d0,
    float d1,
    int sideMask,
    float depth)
{
    s.zonePreviewVerts.clear();
    (void)depth;
    if (r.pts.size() < 2) return;

    float a = std::min(d0, d1);
    float b = std::max(d0, d1);
    float total = r.totalLen();
    int cols = (int)std::floor(total / ZONE_CELL_M);
    if (cols <= 0) return;

    int i0 = (int)std::floor(a / ZONE_CELL_M);
    int i1 = (int)std::ceil(b / ZONE_CELL_M) - 1;
    i0 = std::max(0, i0);
    i1 = std::min(cols - 1, i1);
    if (i1 < i0) return;

    for (int i = i0; i <= i1; ++i) {
        float d = (i + 0.5f) * ZONE_CELL_M;
        glm::vec3 tan;
        glm::vec3 p = r.pointAt(d, tan);
        if (glm::dot(tan, tan) < 1e-6f) continue;

        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0, 1, 0), tan));
        auto drawSide = [&](int side, int sideBit) {
            if (!(sideMask & sideBit)) return;
            glm::vec3 away = right * (float)side;
            for (int row = 0; row < ZONE_DEPTH_CELLS; ++row) {
                float offset = ROAD_HALF_M + (row + 0.5f) * ZONE_CELL_M;
                glm::vec3 center = p + away * offset;
                if (GetWaterAt(s, center) != 0) continue;
                if (ShouldCullForIntersection(s, r.id, center, tan, INTERSECTION_CLEAR_M)) continue;
                AppendOrientedZoneCellQuad(s.zonePreviewVerts, center, tan, away);
            }
        };

        drawSide(-1, 1);
        drawSide(+1, 2);
    }
}

static void AppendRoadInfluencePreview(std::vector<glm::vec3>& out, const Road& r) {
    if (r.pts.size() < 2 || r.cumLen.size() != r.pts.size()) return;
    float total = r.totalLen();
    int cols = (int)std::floor(total / ZONE_CELL_M);
    if (cols <= 0) return;

    for (int i = 0; i < cols; ++i) {
        float d = (i + 0.5f) * ZONE_CELL_M;
        glm::vec3 tan;
        glm::vec3 p = r.pointAt(d, tan);
        if (glm::dot(tan, tan) < 1e-6f) continue;

        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0, 1, 0), tan));
        for (int side : {-1, +1}) {
            glm::vec3 away = right * (float)side;
            for (int row = 0; row < ZONE_DEPTH_CELLS; ++row) {
                float offset = ROAD_HALF_M + (row + 0.5f) * ZONE_CELL_M;
                glm::vec3 center = p + away * offset;
                AppendOrientedZoneCellQuad(out, center, tan, away);
            }
        }
    }
}

static void RebuildLotCells(AppState& s) {
    s.lots.clear();
    s.lotIndicesByChunk.clear();
    if (s.roads.empty()) return;

    const float roadHalf = ROAD_HALF_M;
    const float lotDepth = ZONE_DEPTH_M;
    const float cellLen = ZONE_CELL_M * 2.0f;
    const float desiredClear = 0.0f;
    const float setback = roadHalf + desiredClear + (lotDepth * 0.5f);

    std::unordered_set<uint64_t> occupied;
    auto cellKey = [](int32_t gx, int32_t gz) -> uint64_t {
        return (uint64_t(uint32_t(gx)) << 32) | uint32_t(gz);
    };

    const float dedupCell = 4.0f;
    const float buildableCoverage = 0.85f;

    for (const auto& r : s.roads) {
        if (r.pts.size() < 2) continue;
        float total = r.totalLen();
        for (float d = 0.0f; d + cellLen <= total; d += cellLen) {
            float mid = d + cellLen * 0.5f;
            glm::vec3 tan;
            glm::vec3 base = r.pointAt(mid, tan);
            if (glm::dot(tan, tan) < 1e-6f) continue;
            glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), tan));

            for (int side : {-1, 1}) {
                glm::vec3 center = base + right * float(side) * setback;
                if (!LotRectMeetsGrid(
                        s, center, tan, right, cellLen, lotDepth,
                        ZONE_FLAG_BUILDABLE, ZONE_FLAG_BLOCKED, buildableCoverage)) {
                    continue;
                }

                int32_t gx = (int32_t)std::floor(center.x / dedupCell);
                int32_t gz = (int32_t)std::floor(center.z / dedupCell);
                uint64_t k = cellKey(gx, gz);
                if (occupied.find(k) != occupied.end()) continue;

                LotCell c;
                c.roadId = r.id;
                c.side = side;
                c.d0 = d;
                c.d1 = d + cellLen;
                c.center = center;
                c.forward = glm::normalize(tan);
                c.right = right;
                ZoneType zt = ZoneType::Residential;
                c.zoned = IsLotZoned(s, c, zt);
                c.zoneType = zt;

                occupied.insert(k);
                int idx = (int)s.lots.size();
                s.lots.push_back(c);
                uint64_t ck = PackChunk(ChunkFromPosXZ(center).cx, ChunkFromPosXZ(center).cz);
                s.lotIndicesByChunk[ck].push_back(idx);
            }
        }
    }
}

static void BuildRoadPreviewMesh(AppState& s, const glm::vec3& a, const glm::vec3& b) {
    const float roadWidth = ROAD_WIDTH_M;
    const float y = 0.05f;

    glm::vec3 dir = b - a;
    dir.y = 0.0f;
    float len = std::sqrt(dir.x*dir.x + dir.z*dir.z);
    if (len < 1e-3f) return;
    dir /= len;

    glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), dir));
    glm::vec3 off = right * (roadWidth * 0.5f);

    glm::vec3 aL = a - off; aL.y = y;
    glm::vec3 aR = a + off; aR.y = y;
    glm::vec3 bL = b - off; bL.y = y;
    glm::vec3 bR = b + off; bR.y = y;

    s.zonePreviewVerts.push_back(aL);
    s.zonePreviewVerts.push_back(aR);
    s.zonePreviewVerts.push_back(bR);

    s.zonePreviewVerts.push_back(aL);
    s.zonePreviewVerts.push_back(bR);
    s.zonePreviewVerts.push_back(bL);
}

static glm::vec3 ApplyAssetScale(const AssetCatalog& assets, AssetId assetId, const glm::vec3& baseSize) {
    const AssetDef* def = assets.find(assetId);
    if (!def) return baseSize;
    glm::vec3 scaled = def->meshRelPath.empty() ? baseSize : def->defaultScale;
    if (scaled.x <= 0.0f || scaled.y <= 0.0f || scaled.z <= 0.0f) return baseSize;
    return scaled;
}

static glm::vec2 GetAssetFootprint(const AssetCatalog& assets, AssetId assetId, const glm::vec2& fallback) {
    const AssetDef* def = assets.find(assetId);
    if (!def) return fallback;
    if (def->meshRelPath.empty()) return fallback;
    if (def->footprintM.x > 0.0f && def->footprintM.y > 0.0f) return def->footprintM;
    return fallback;
}

[[maybe_unused]] static bool LotRectMeetsRoadBand(
    const glm::vec3& center,
    const glm::vec3& forward,
    const glm::vec3& right,
    float width,
    float depth,
    float roadHalf,
    float desiredClear,
    float lotDepth,
    const std::vector<Road>& roads,
    const Road* extraRoad)
{
    int nx = std::max(1, (int)std::ceil(width / ZONE_CELL_M));
    int nz = std::max(1, (int)std::ceil(depth / ZONE_CELL_M));
    float stepX = width / (float)nx;
    float stepZ = depth / (float)nz;
    float halfW = width * 0.5f;
    float halfD = depth * 0.5f;

    auto minDistSqToRoads = [&](const glm::vec3& p) -> float {
        float best = std::numeric_limits<float>::max();
        for (const auto& r : roads) {
            if (r.pts.size() < 2) continue;
            float dAlong; glm::vec3 tan;
            float distSq = ClosestDistanceAlongRoadSq(r, p, dAlong, tan);
            best = std::min(best, distSq);
        }
        if (extraRoad && extraRoad->pts.size() >= 2) {
            float dAlong; glm::vec3 tan;
            float distSq = ClosestDistanceAlongRoadSq(*extraRoad, p, dAlong, tan);
            best = std::min(best, distSq);
        }
        return best;
    };

    for (int iz = 0; iz < nz; iz++) {
        float v = -halfD + (iz + 0.5f) * stepZ;
        for (int ix = 0; ix < nx; ix++) {
            float u = -halfW + (ix + 0.5f) * stepX;
            glm::vec3 p = center + right * u + forward * v;
            float distSq = minDistSqToRoads(p);
            float distEdge = std::sqrt(distSq) - roadHalf;
            if (distEdge < desiredClear || distEdge > desiredClear + lotDepth) return false;
        }
    }
    return true;
}

[[maybe_unused]] static void AppendLotGridPreviewForRoad(std::vector<glm::vec3>& out, const Road& r, const std::vector<Road>& otherRoads) {
    if (r.pts.size() < 2 || r.cumLen.size() != r.pts.size()) return;

    const float roadHalf = ROAD_HALF_M;
    const float lotDepth = ZONE_DEPTH_M;
    const float cellLen = ZONE_CELL_M * 2.0f;
    const float desiredClear = 0.0f;
    const float setback = roadHalf + desiredClear + (lotDepth * 0.5f);

    float total = r.totalLen();
    for (float d = 0.0f; d + cellLen <= total; d += cellLen) {
        float mid = d + cellLen * 0.5f;
        glm::vec3 tan;
        glm::vec3 base = r.pointAt(mid, tan);
        if (glm::dot(tan, tan) < 1e-6f) continue;
        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), tan));

        for (int side : {-1, 1}) {
            glm::vec3 center = base + right * float(side) * setback;
            if (!LotRectMeetsRoadBand(center, tan, right, cellLen, lotDepth, roadHalf, desiredClear, lotDepth, otherRoads, &r)) {
                continue;
            }
            float lotWidth = std::max(6.0f, cellLen);
            AppendLotOverlayQuad(out, center, tan, right, lotWidth, lotDepth);
        }
    }
}

static void RebuildHousesFromLots(AppState& s, const AssetCatalog& assets, bool animate, float nowSec) {
    s.houseStatic.clear();
    s.houseAnim.clear();
    s.houseStaticByChunk.clear();
    s.buildingChunks.clear();
    s.dirtyBuildingChunks.clear();

    const float roadHalf = ROAD_HALF_M;
    const float desiredClear = 0.0f; // matches buildable band start
    const float lotDepth = ZONE_DEPTH_M;
    const AssetId residentialAsset = assets.resolveCategoryAsset(ZoneTypeCategory(ZoneType::Residential));
    const AssetId commercialAsset = assets.resolveCategoryAsset(ZoneTypeCategory(ZoneType::Commercial));
    const AssetId industrialAsset = assets.resolveCategoryAsset(ZoneTypeCategory(ZoneType::Industrial));
    const AssetId officeAsset = assets.resolveCategoryAsset(ZoneTypeCategory(ZoneType::Office));

    std::unordered_set<uint64_t> occupied;
    auto cellKey = [](int32_t gx, int32_t gz) -> uint64_t {
        return (uint64_t(uint32_t(gx)) << 32) | uint32_t(gz);
    };
    auto isOccupied = [&](const glm::vec3& pos) {
        const float cell = 6.0f; // coarse grid to prevent overlapping houses
        int32_t gx = (int32_t)std::floor(pos.x / cell);
        int32_t gz = (int32_t)std::floor(pos.z / cell);
        return occupied.find(cellKey(gx, gz)) != occupied.end();
    };
    auto markOccupied = [&](const glm::vec3& pos) {
        const float cell = 6.0f;
        int32_t gx = (int32_t)std::floor(pos.x / cell);
        int32_t gz = (int32_t)std::floor(pos.z / cell);
        occupied.insert(cellKey(gx, gz));
    };
    struct PlacedHouse {
        glm::vec3 pos{};
        float radius = 0.0f;
    };
    std::vector<PlacedHouse> placed;
    std::unordered_map<uint64_t, std::vector<int>> placedByCell;
    const float placementCell = 8.0f;
    auto placeCellKey = [](int32_t gx, int32_t gz) -> uint64_t {
        return (uint64_t(uint32_t(gx)) << 32) | uint32_t(gz);
    };
    auto addPlaced = [&](const glm::vec3& pos, float radius) {
        PlacedHouse ph{pos, radius};
        int idx = (int)placed.size();
        placed.push_back(ph);
        int32_t gx = (int32_t)std::floor(pos.x / placementCell);
        int32_t gz = (int32_t)std::floor(pos.z / placementCell);
        placedByCell[placeCellKey(gx, gz)].push_back(idx);
    };
    auto canPlace = [&](const glm::vec3& pos, float radius) -> bool {
        int32_t gx = (int32_t)std::floor(pos.x / placementCell);
        int32_t gz = (int32_t)std::floor(pos.z / placementCell);
        int range = (int)std::ceil(radius / placementCell) + 1;
        float minDist = radius + 0.5f;
        for (int dz = -range; dz <= range; dz++) {
            for (int dx = -range; dx <= range; dx++) {
                auto it = placedByCell.find(placeCellKey(gx + dx, gz + dz));
                if (it == placedByCell.end()) continue;
                for (int idx : it->second) {
                    const auto& other = placed[idx];
                    float minPair = minDist + other.radius;
                    glm::vec3 d = pos - other.pos;
                    if (glm::dot(d, d) < minPair * minPair) return false;
                }
            }
        }
        return true;
    };

    auto minCenterlineClearSq = [&](const glm::vec3& pos) -> float {
        float best = std::numeric_limits<float>::max();
        for (const auto& other : s.roads) {
            if (other.pts.size() < 2) continue;
            float dAlong; glm::vec3 tan;
            float distSq = ClosestDistanceAlongRoadSq(other, pos, dAlong, tan);
            best = std::min(best, distSq);
        }
        return best;
    };

    for (const auto& c : s.lots) {
        if (!c.zoned) continue;
        if (GetZoneFlagsAt(s, c.center) & ZONE_FLAG_BLOCKED) continue;

        ZoneType lotType = c.zoneType;
        AssetId assetId = residentialAsset;
        switch (lotType) {
            case ZoneType::Commercial: assetId = commercialAsset; break;
            case ZoneType::Industrial: assetId = industrialAsset; break;
            case ZoneType::Office: assetId = officeAsset; break;
            default: assetId = residentialAsset; break;
        }

        glm::vec3 baseSize = BaseSizeForZone(lotType);
        glm::vec3 houseSize = ApplyAssetScale(assets, assetId, baseSize);
        glm::vec2 footprint = GetAssetFootprint(assets, assetId, glm::vec2(baseSize.x, baseSize.z));
        float alignedAlong = std::ceil(footprint.x / ZONE_CELL_M) * ZONE_CELL_M;
        float alignedDepth = std::ceil(footprint.y / ZONE_CELL_M) * ZONE_CELL_M;
        alignedAlong = std::max(alignedAlong, ZONE_CELL_M);
        alignedDepth = std::max(alignedDepth, ZONE_CELL_M);
        if (alignedDepth > lotDepth) continue;

        float radius = 0.5f * std::sqrt(alignedAlong * alignedAlong + alignedDepth * alignedDepth);

        glm::vec3 pos = c.center;
        pos.y = houseSize.y * 0.5f;

        float distSq = minCenterlineClearSq(pos);
        float clearFromEdge = std::sqrt(distSq) - roadHalf; // distance from nearest road edge
        if (clearFromEdge < desiredClear) continue; // too close to any road (intersections)
        if (isOccupied(pos)) continue; // avoid double builds/overlap
        if (!canPlace(pos, radius)) continue;

        glm::vec3 up(0,1,0);
        glm::vec3 facing = glm::normalize(-float(c.side) * c.right); // face toward road
        glm::vec3 basisRight = glm::normalize(glm::cross(up, facing));
        glm::mat4 R(1.0f);
        R[0] = glm::vec4(basisRight, 0.0f);
        R[1] = glm::vec4(up, 0.0f);
        R[2] = glm::vec4(facing, 0.0f);

        uint32_t hx = (uint32_t)std::llround(pos.x * 10.0);
        uint32_t hz = (uint32_t)std::llround(pos.z * 10.0);
        uint32_t seed = Hash32(hx ^ (hz * 1664525U) ^ (uint32_t)(c.roadId * 131071U) ^ (c.side < 0 ? 0x9e3779b9U : 0U));
        float yaw = std::atan2(facing.x, facing.z);
        if (animate) {
            float jitter = (seed % 120) / 1000.0f; // 0..0.119 sec
            s.houseAnim.push_back({pos, nowSec + jitter, facing, assetId, houseSize, seed});
        } else {
            glm::mat4 M(1.0f);
            M = glm::translate(M, pos);
            M = M * R;
            M = glm::scale(M, houseSize);
            s.houseStatic.push_back(M);
        }
        // Chunked storage for rendering
        glm::mat4 Ms(1.0f);
        Ms = glm::translate(Ms, pos);
        Ms = Ms * R;
        Ms = glm::scale(Ms, houseSize);
        ChunkCoord cc = ChunkFromPosXZ(pos);
        uint64_t ckey = PackChunk(cc.cx, cc.cz);
        s.houseStaticByChunk[ckey].push_back(Ms);
        BuildingInstance inst;
        inst.asset = assetId;
        inst.localPos = pos;
        inst.yaw = yaw;
        inst.scale = houseSize;
        inst.seed = seed;
        s.buildingChunks[ckey].instancesByAsset[assetId].push_back(inst);
        s.dirtyBuildingChunks.insert(ckey);
        markOccupied(pos);
        addPlaced(pos, radius);
    }
}

static bool SaveToJsonFile(const AppState& s, const AssetCatalog& assets, const std::string& path) {
    json j;
    j["version"] = 1;
    j["nextRoadId"] = s.nextRoadId;
    j["nextZoneId"] = s.nextZoneId;
    json assetMap = json::object();
    for (const auto& kv : assets.assets()) {
        assetMap[std::to_string(kv.first)] = kv.second.idStr;
    }
    j["assetIdToString"] = assetMap;

    j["roads"] = json::array();
    for (const auto& r : s.roads) {
        json jr;
        jr["id"] = r.id;
        jr["pts"] = json::array();
        for (auto& p : r.pts) {
            jr["pts"].push_back({p.x, p.y, p.z});
        }
        j["roads"].push_back(jr);
    }

    j["zones"] = json::array();
    for (const auto& z : s.zones) {
        json jz;
        jz["id"] = z.id;
        jz["roadId"] = z.roadId;
        jz["d0"] = z.d0;
        jz["d1"] = z.d1;
        jz["sideMask"] = z.sideMask;
        jz["zoneType"] = (int)z.type;
        jz["depth"] = ZONE_DEPTH_M;
        j["zones"].push_back(jz);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << j.dump(2);
    return true;
}

// Placeholder chunk save/load (binary) for zone/building chunks.
static bool SaveChunkBin(const AppState&, uint64_t, const std::string&) { return true; }
static bool LoadChunkBin(AppState&, uint64_t, const std::string&) { return true; }

static bool LoadFromJsonFile(AppState& s, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    json j;
    in >> j;

    int ver = j.value("version", 0);
    if (ver != 1) return false;

    s.nextRoadId = j.value("nextRoadId", 1);
    s.nextZoneId = j.value("nextZoneId", 1);

    s.roads.clear();
    s.zones.clear();

    for (auto& jr : j["roads"]) {
        Road r;
        r.id = jr.value("id", 0);
        for (auto& jp : jr["pts"]) {
            glm::vec3 p;
            p.x = jp[0].get<float>();
            p.y = jp[1].get<float>();
            p.z = jp[2].get<float>();
            p.y = 0.0f;
            r.pts.push_back(p);
        }
        r.rebuildCum();
        s.roads.push_back(std::move(r));
    }

    for (auto& jz : j["zones"]) {
        ZoneStrip z;
        z.id = jz.value("id", 0);
        z.roadId = jz.value("roadId", 0);
        z.d0 = jz.value("d0", 0.0f);
        z.d1 = jz.value("d1", 0.0f);
        z.sideMask = jz.value("sideMask", 3);
        int typeVal = jz.value("zoneType", 0);
        typeVal = (int)Clamp((float)typeVal, 0.0f, 3.0f);
        z.type = (ZoneType)typeVal;
        z.depth = ZONE_DEPTH_M;
        s.zones.push_back(z);
    }

    s.roadsDirty = true;
    s.zonesDirty = true;
    s.housesDirty = true;
    return true;
}

// Tool states
enum class Mode { Road, Zone, Unzone };

struct RoadTool {
    bool drawing = false;
    bool extending = false;
    bool extendAtStart = false;
    int extendRoadId = -1;

    std::vector<glm::vec3> tempPts;

    int selectedRoadId = -1;
    int selectedPointIndex = -1;
    bool movingPoint = false;
    glm::vec3 moveOld{};
};

struct ZoneTool {
    bool dragging = false;
    int roadId = -1;
    float startD = 0.0f;
    float endD = 0.0f;

    bool hoverValid = false;
    int hoverRoadId = -1;
    float hoverD = 0.0f;

    int sideMask = 3;   // 1 left, 2 right, 3 both
    ZoneType type = ZoneType::Residential;
    float depth = ZONE_DEPTH_M;
    float pickRadius = 12.0f;

    // For overlay visualization of zoned road spans
    std::vector<glm::vec3> overlayVerts;
};

static float ClosestDistanceAlongRoadSq(const Road& r, const glm::vec3& p, float& outAlong, glm::vec3& outTan) {
    float bestDistSq = 1e30f;
    float bestAlong = 0.0f;
    glm::vec3 bestTan(1,0,0);

    if (r.pts.size() < 2) {
        outAlong = 0.0f;
        outTan = bestTan;
        return bestDistSq;
    }

    for (size_t i = 0; i + 1 < r.pts.size(); i++) {
        glm::vec3 a = r.pts[i];
        glm::vec3 b = r.pts[i+1];
        glm::vec3 c;
        float t = ClosestParamOnSegmentXZ(p, a, b, c);

        glm::vec2 d(p.x - c.x, p.z - c.z);
        float distSq = d.x*d.x + d.y*d.y;

        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            float segLen = LenXZ(a, b);
            float along = (i < r.cumLen.size() ? r.cumLen[i] : 0.0f) + t * segLen;
            bestAlong = along;

            glm::vec3 dir = b - a;
            dir.y = 0.0f;
            float l = std::sqrt(dir.x*dir.x + dir.z*dir.z);
            if (l > 1e-6f) dir /= l;
            bestTan = dir;
        }
    }

    outAlong = bestAlong;
    outTan = bestTan;
    return bestDistSq;
}

int main(int, char**) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* window = SDL_CreateWindow(
        "City Painter Prototype (Phase 1)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (!window) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext glctx = SDL_GL_CreateContext(window);
    if (!glctx) {
        SDL_Log("SDL_GL_CreateContext failed: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_GL_MakeCurrent(window, glctx);
    SDL_GL_SetSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
        SDL_Log("gladLoadGLLoader failed");
        SDL_GL_DeleteContext(glctx);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    Renderer renderer;
    if (!renderer.init()) {
        SDL_Log("Renderer init failed");
        SDL_GL_DeleteContext(glctx);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window, glctx);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    AssetCatalog assets;
    assets.loadAll("assets");

    MeshCache meshCache;
    if (!meshCache.init()) {
        SDL_Log("MeshCache init failed");
    }

    AppState state;
    CommandStack cmds;

    Camera cam;
    Mode mode = Mode::Road;
    RoadTool roadTool;
    ZoneTool zoneTool;

    // snapping + UX settings
    bool gridSnap = true;
    float gridSize = 2.0f;

    bool angleSnap = true;

    bool endpointSnap = true;
    float endpointSnapRadius = 10.0f;

    float roadPointPickRadius = 6.0f;

    // Save/load UI
    char savePath[260] = "save.json";
    char waterMapPath[260] = "assets/maps/water_8192.png";
    float waterThreshold = 0.5f;
    float timeOfDayHours = 12.0f;
    std::string statusText;
    MinimapState minimap;

    bool running = true;
    bool rmbDown = false;
    bool mmbDown = false;
    int winW = 1280, winH = 720;
    SDL_GetWindowSize(window, &winW, &winH);
    renderer.resize(winW, winH);

    uint64_t perfFreq = SDL_GetPerformanceFrequency();
    uint64_t lastCounter = SDL_GetPerformanceCounter();

    auto applySnaps = [&](glm::vec3 raw, const glm::vec3* prevOpt) {
        glm::vec3 p = raw;
        if (gridSnap) p = SnapToGridXZ(p, gridSize);
        if (angleSnap && prevOpt) p = SnapAngle15FromPrev(*prevOpt, p);

        if (endpointSnap) {
            glm::vec3 ep;
            int rid;
            bool isStart;
            if (SnapToAnyEndpoint(state.roads, p, endpointSnapRadius, ep, rid, isStart)) {
                p = ep;
            }
        }
        p.y = 0.0f;
        return p;
    };

    auto startRoadDraw = [&](glm::vec3 hit) {
        roadTool.tempPts.clear();

        // Anchor at an existing endpoint if near one, but always create a new road (do not modify existing)
        glm::vec3 p0 = hit;
        bool anchoredToEndpoint = false;
        if (endpointSnap) {
            glm::vec3 ep; int rid; bool isStart;
            if (SnapToAnyEndpoint(state.roads, hit, endpointSnapRadius, ep, rid, isStart)) {
                p0 = ep;
                anchoredToEndpoint = true;
            }
        }
        if (!anchoredToEndpoint) {
            p0 = applySnaps(p0, nullptr);
        }
        roadTool.tempPts.push_back(p0);

        roadTool.extending = false;
        roadTool.extendAtStart = false;
        roadTool.extendRoadId = -1;

        roadTool.drawing = true;
    };

    auto finishRoadDraw = [&]() {
        if (!roadTool.drawing) return;

        float segLen = (roadTool.tempPts.size() >= 2) ? LenXZ(roadTool.tempPts.front(), roadTool.tempPts.back()) : 0.0f;
        bool hasLine = (roadTool.tempPts.size() >= 2) && (segLen >= 1.0f);

        // Must have at least 2 points for a new road
        if (!roadTool.extending) {
            if (hasLine) {
                Road r;
                r.id = state.nextRoadId++;
                r.pts = roadTool.tempPts;
                r.rebuildCum();
                cmds.exec(state, std::make_unique<CmdAddRoad>(r));
                statusText = "Road created.";
            } else {
                statusText = "Road canceled (too short).";
            }
        } else {
            // Extend existing road by adding points (excluding the first anchor point)
            if (hasLine) {
                std::vector<glm::vec3> added(roadTool.tempPts.begin() + 1, roadTool.tempPts.end());
                cmds.exec(state, std::make_unique<CmdExtendRoad>(roadTool.extendRoadId, added, roadTool.extendAtStart));
                statusText = "Road extended.";
            } else {
                statusText = "Extend canceled (too short).";
            }
        }

        roadTool.drawing = false;
        roadTool.extending = false;
        roadTool.extendRoadId = -1;
        roadTool.tempPts.clear();
    };

    auto updateZoneHover = [&](const glm::vec3& hit) {
        zoneTool.hoverValid = false;
        zoneTool.hoverRoadId = -1;
        zoneTool.hoverD = 0.0f;

        float bestSq = zoneTool.pickRadius * zoneTool.pickRadius;
        int bestRoad = -1;
        float bestD = 0.0f;

        for (const auto& r : state.roads) {
            if (r.pts.size() < 2 || r.cumLen.size() != r.pts.size()) continue;
            float dAlong; glm::vec3 tan;
            float distSq = ClosestDistanceAlongRoadSq(r, hit, dAlong, tan);
            if (distSq < bestSq) {
                bestSq = distSq;
                bestRoad = r.id;
                bestD = dAlong;
            }
        }

        if (bestRoad != -1) {
            zoneTool.hoverValid = true;
            zoneTool.hoverRoadId = bestRoad;
            zoneTool.hoverD = bestD;
        }
    };

    while (running) {
        uint64_t counter = SDL_GetPerformanceCounter();
        double dt = double(counter - lastCounter) / double(perfFreq);
        lastCounter = counter;
        float fdt = (float)dt;

        float nowSec = SDL_GetTicks() / 1000.0f;

        // Movement (ignore when typing in ImGui)
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard) {
            const Uint8* ks = SDL_GetKeyboardState(nullptr);
            glm::vec3 pos = cam.position();
            glm::vec3 forward = glm::normalize(glm::vec3(cam.target.x - pos.x, 0.0f, cam.target.z - pos.z));
            glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0,1,0)));

            float panSpeed = 250.0f;
            if (ks[SDL_SCANCODE_LSHIFT]) panSpeed *= 2.0f;

            if (ks[SDL_SCANCODE_W]) cam.target += forward * panSpeed * fdt;
            if (ks[SDL_SCANCODE_S]) cam.target -= forward * panSpeed * fdt;
            if (ks[SDL_SCANCODE_D]) cam.target += right   * panSpeed * fdt;
            if (ks[SDL_SCANCODE_A]) cam.target -= right   * panSpeed * fdt;
        }

        int mx, my;
        SDL_GetMouseState(&mx, &my);

        glm::vec3 renderOrigin = ComputeRenderOrigin(cam.target);

        float aspect = (winH > 0) ? (float)winW / (float)winH : 1.0f;
        float nearClip = Clamp(cam.distance * 0.05f, 20.0f, 300.0f);
        float farClip  = Clamp(cam.distance * 60.0f, 5000.0f, 120000.0f);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, nearClip, farClip);

        glm::vec3 eye = cam.position() - renderOrigin;
        glm::vec3 tgt = cam.target - renderOrigin;
        glm::mat4 view = glm::lookAt(eye, tgt, glm::vec3(0,1,0));
        glm::mat4 viewProj = proj * view;
        glm::mat4 viewSky = glm::mat4(glm::mat3(view));
        glm::mat4 viewProjSky = proj * viewSky;

        LightingParams lighting = EvaluateTimeOfDay(timeOfDayHours);
        float shadowRadius = Clamp(cam.distance * 2.4f, 400.0f, 9000.0f);
        glm::mat4 lightViewProj = BuildDirectionalLightMatrix(tgt, shadowRadius, lighting.sunDir);

        glm::vec3 mouseHitRel;
        bool hasHit = ScreenToGroundHit(mx, my, winW, winH, view, proj, mouseHitRel);
        glm::vec3 mouseHit = mouseHitRel + renderOrigin;

        // Visible chunks around camera
        ChunkCoord camChunk = ChunkFromPosXZ(cam.target);
        const int viewRadius = 5;
        std::vector<uint64_t> visibleChunks;
        visibleChunks.reserve((2 * viewRadius + 1) * (2 * viewRadius + 1));
        for (int dz = -viewRadius; dz <= viewRadius; ++dz) {
            for (int dx = -viewRadius; dx <= viewRadius; ++dx) {
                visibleChunks.push_back(PackChunk(camChunk.cx + dx, camChunk.cz + dz));
            }
        }
        std::unordered_set<uint64_t> visibleChunkSet(visibleChunks.begin(), visibleChunks.end());

        // Update hover for zoning/unzoning
        if ((mode == Mode::Zone || mode == Mode::Unzone) && hasHit) {
            updateZoneHover(mouseHit);
        }

        // Events
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);

            if (e.type == SDL_QUIT) running = false;

            if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                winW = e.window.data1;
                winH = e.window.data2;
                renderer.resize(winW, winH);
            }

            if (e.type == SDL_MOUSEWHEEL && !io.WantCaptureMouse) {
                float wheel = (float)e.wheel.y;
                cam.distance *= std::pow(0.90f, wheel);
                cam.distance = Clamp(cam.distance, 30.0f, 4000.0f);
            }

            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT && !io.WantCaptureMouse) {
                rmbDown = true;
                SDL_SetRelativeMouseMode(SDL_TRUE);
            }
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT) {
                rmbDown = false;
                if (!mmbDown) SDL_SetRelativeMouseMode(SDL_FALSE);
            }

            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_MIDDLE && !io.WantCaptureMouse) {
                mmbDown = true;
                SDL_SetRelativeMouseMode(SDL_TRUE);
            }
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_MIDDLE) {
                mmbDown = false;
                if (!rmbDown) SDL_SetRelativeMouseMode(SDL_FALSE);
            }
            if (e.type == SDL_MOUSEMOTION && rmbDown && !io.WantCaptureMouse) {
                cam.yawRad += (float)e.motion.xrel * 0.004f;
            }
            if (e.type == SDL_MOUSEMOTION && mmbDown && !io.WantCaptureMouse) {
                cam.pitchDeg = Clamp(cam.pitchDeg - (float)e.motion.yrel * 0.25f, 15.0f, 85.0f);
            }

            if (e.type == SDL_KEYDOWN && !e.key.repeat && !io.WantCaptureKeyboard) {
                SDL_Keycode k = e.key.keysym.sym;

                if (k == SDLK_ESCAPE) running = false;

                if (k == SDLK_1) { mode = Mode::Road; statusText = "Road mode."; }
                if (k == SDLK_2) { mode = Mode::Zone; zoneTool.type = ZoneType::Residential; statusText = "Zone: Residential."; }
                if (k == SDLK_3) { mode = Mode::Zone; zoneTool.type = ZoneType::Commercial; statusText = "Zone: Commercial."; }
                if (k == SDLK_4) { mode = Mode::Zone; zoneTool.type = ZoneType::Industrial; statusText = "Zone: Industrial."; }
                if (k == SDLK_5) { mode = Mode::Zone; zoneTool.type = ZoneType::Office; statusText = "Zone: Office."; }
                if (k == SDLK_6) { mode = Mode::Unzone; statusText = "Unzone mode."; }

                if (k == SDLK_g) { gridSnap = !gridSnap; statusText = gridSnap ? "Grid snap ON" : "Grid snap OFF"; }
                if (k == SDLK_h) { angleSnap = !angleSnap; statusText = angleSnap ? "Angle snap ON" : "Angle snap OFF"; }

                if (k == SDLK_v) {
                    // cycle zoning side
                    if (zoneTool.sideMask == 3) zoneTool.sideMask = 1;
                    else if (zoneTool.sideMask == 1) zoneTool.sideMask = 2;
                    else zoneTool.sideMask = 3;
                }

                bool ctrl = (SDL_GetModState() & KMOD_CTRL) != 0;
                bool shift = (SDL_GetModState() & KMOD_SHIFT) != 0;

                if (ctrl && k == SDLK_z && !shift) {
                    cmds.doUndo(state);
                    statusText = "Undo.";
                }
                if ((ctrl && k == SDLK_y) || (ctrl && shift && k == SDLK_z)) {
                    cmds.doRedo(state);
                    statusText = "Redo.";
                }

                if (ctrl && k == SDLK_s) {
                    if (SaveToJsonFile(state, assets, savePath)) statusText = "Saved.";
                    else statusText = "Save failed.";
                }

                if (ctrl && k == SDLK_o) {
                    if (LoadFromJsonFile(state, savePath)) {
                        cmds.clear();
                        statusText = "Loaded.";
                    } else statusText = "Load failed.";
                }

                if (mode == Mode::Road) {
                    if ((k == SDLK_DELETE || k == SDLK_BACKSPACE) && roadTool.selectedRoadId != -1 && roadTool.selectedPointIndex != -1) {
                        cmds.exec(state, std::make_unique<CmdDeleteRoadPoint>(roadTool.selectedRoadId, roadTool.selectedPointIndex));
                        statusText = "Point deleted.";
                        roadTool.selectedPointIndex = -1;
                    }
                }
            }

            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT && !io.WantCaptureMouse) {
                if (!hasHit) break;

                if (mode == Mode::Road) {
                    // If clicking near a road point: start moving interior points, but endpoints extend the road.
                    int rid, pi;
                    if (PickRoadPoint(state.roads, mouseHit, roadPointPickRadius, rid, pi)) {
                        int idx = FindRoadIndexById(state.roads, rid);
                        bool isEndpoint = (idx >= 0) && (pi == 0 || pi == (int)state.roads[idx].pts.size() - 1);
                        if (!isEndpoint) {
                            roadTool.selectedRoadId = rid;
                            roadTool.selectedPointIndex = pi;
                            roadTool.movingPoint = true;

                            if (idx >= 0) roadTool.moveOld = state.roads[idx].pts[pi];

                            statusText = "Moving point (drag).";
                        } else {
                            // Extend the existing road from this endpoint.
                            roadTool.selectedRoadId = -1;
                            roadTool.selectedPointIndex = -1;
                            startRoadDraw(state.roads[idx].pts[pi]);
                            roadTool.extending = true;
                            roadTool.extendRoadId = rid;
                            roadTool.extendAtStart = (pi == 0);
                            statusText = "Extending road.";
                        }
                    } else {
                        roadTool.selectedRoadId = -1;
                        roadTool.selectedPointIndex = -1;
                        startRoadDraw(mouseHit);
                    }
                } else if (mode == Mode::Zone) {
                    // Zone mode: must start near a road
                    if (!zoneTool.hoverValid) {
                        statusText = "Invalid: must start zoning near a road.";
                        break;
                    }
                    zoneTool.dragging = true;
                    zoneTool.roadId = zoneTool.hoverRoadId;
                    zoneTool.startD = zoneTool.hoverD;
                    zoneTool.endD = zoneTool.hoverD;
                } else if (mode == Mode::Unzone) {
                    if (!zoneTool.hoverValid) {
                        statusText = "Invalid: click near a road to unzone.";
                        break;
                    }
                    int rid = zoneTool.hoverRoadId;
                    std::vector<ZoneStrip> removed;
                    for (const auto& z : state.zones) if (z.roadId == rid) removed.push_back(z);
                    if (removed.empty()) {
                        statusText = "No zones to clear.";
                        break;
                    }
                    cmds.exec(state, std::make_unique<CmdClearZonesForRoad>(rid, removed));
                    statusText = "Zones cleared.";
                }
            }

            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT && !io.WantCaptureMouse) {
                if (mode == Mode::Road) {
                    if (roadTool.movingPoint) {
                        // Commit move
                        int idx = FindRoadIndexById(state.roads, roadTool.selectedRoadId);
                        if (idx >= 0 && roadTool.selectedPointIndex >= 0) {
                            glm::vec3 newPos = state.roads[idx].pts[roadTool.selectedPointIndex];
                            cmds.exec(state, std::make_unique<CmdMoveRoadPoint>(roadTool.selectedRoadId, roadTool.selectedPointIndex, roadTool.moveOld, newPos));
                            statusText = "Point move committed.";
                        }
                        roadTool.movingPoint = false;
                    }
                    finishRoadDraw();
                } else {
                    if (zoneTool.dragging) {
        // Commit zone strip
        ZoneStrip z;
        z.id = state.nextZoneId++;
        z.roadId = zoneTool.roadId;
        z.d0 = zoneTool.startD;
        z.d1 = zoneTool.endD;
        z.sideMask = zoneTool.sideMask;
        z.type = zoneTool.type;
        z.depth = ZONE_DEPTH_M;

        int ridx = FindRoadIndexById(state.roads, z.roadId);
        if (ridx >= 0) {
            float lo = std::min(zoneTool.startD, zoneTool.endD);
            float hi = std::max(zoneTool.startD, zoneTool.endD);
            float total = state.roads[ridx].totalLen();
            int cols = (int)std::floor(total / ZONE_CELL_M);
            if (cols > 0) {
                int i0 = (int)std::floor(lo / ZONE_CELL_M);
                int i1 = (int)std::ceil(hi / ZONE_CELL_M) - 1;
                i0 = std::max(0, std::min(cols - 1, i0));
                i1 = std::max(0, std::min(cols - 1, i1));
                if (i1 < i0) {
                    int tmp = i0;
                    i0 = i1;
                    i1 = tmp;
                }
                z.d0 = i0 * ZONE_CELL_M;
                z.d1 = (i1 + 1) * ZONE_CELL_M;
            }
        }

        if (ZoneOverlapsExisting(state, z.roadId, z.d0, z.d1)) {
            statusText = "Already zoned here.";
        } else {
            cmds.exec(state, std::make_unique<CmdAddZone>(z));
                            statusText = "Zone committed.";
                        }
                        zoneTool.dragging = false;
                        zoneTool.roadId = -1;
                    }
                }
            }
        }

        // Continuous actions
        if (!io.WantCaptureMouse) {
            // Road drawing: update preview end while holding LMB
            if (mode == Mode::Road && roadTool.drawing && hasHit) {
                if (!roadTool.tempPts.empty()) {
                    glm::vec3 anchor = roadTool.tempPts.front();
                    glm::vec3 p = applySnaps(mouseHit, &anchor);
                    if (roadTool.tempPts.size() == 1) roadTool.tempPts.push_back(p);
                    else roadTool.tempPts[1] = p;
                }
            }

            // Road point moving: drag selected point
            if (mode == Mode::Road && roadTool.movingPoint && hasHit) {
                int idx = FindRoadIndexById(state.roads, roadTool.selectedRoadId);
                if (idx >= 0 && roadTool.selectedPointIndex >= 0 && roadTool.selectedPointIndex < (int)state.roads[idx].pts.size()) {
                    glm::vec3 p = mouseHit;
                    if (gridSnap) p = SnapToGridXZ(p, gridSize);

                    if (endpointSnap) {
                        glm::vec3 ep; int rid; bool isStart;
                        if (SnapToAnyEndpoint(state.roads, p, endpointSnapRadius, ep, rid, isStart)) p = ep;
                    }

                    state.roads[idx].pts[roadTool.selectedPointIndex] = p;
                    state.roads[idx].rebuildCum();
                    state.roadsDirty = true;
                    state.housesDirty = true;
                }
            }

            // Zone drag updates end distance along same road
            if (mode == Mode::Zone && zoneTool.dragging && hasHit) {
                int ridx = FindRoadIndexById(state.roads, zoneTool.roadId);
                if (ridx >= 0) {
                    float dAlong; glm::vec3 tan;
                    (void)ClosestDistanceAlongRoadSq(state.roads[ridx], mouseHit, dAlong, tan);
                    zoneTool.endD = dAlong;
                }
            }
        }

        // Rebuild roads/zones/lot grid if dirty
        if (state.roadsDirty || state.zonesDirty) {
            for (auto& r : state.roads) {
                if (r.cumLen.size() != r.pts.size()) r.rebuildCum();
            }
            if (state.roadsDirty) {
                RebuildAllRoadMesh(state);
            }
            RebuildZoneGrid(state);
            RebuildLotCells(state);
            state.overlayDirty = true;
            state.roadsDirty = false;
            state.zonesDirty = false;
            state.housesDirty = true;
        }

        // Rebuild houses if zones changed
        if (state.housesDirty) {
            bool animate = true; // animate after zone/road edits for now
            RebuildHousesFromLots(state, assets, animate, nowSec);
            state.housesDirty = false;
        }

        // House animation step (move finished anim houses into static instances)
        std::vector<HouseInstanceGPU> animInstances;
        animInstances.reserve(state.houseAnim.size());

        std::vector<HouseAnim> still;
        still.reserve(state.houseAnim.size());

        for (const auto& h : state.houseAnim) {
            float t = (nowSec - h.spawnTime) / 0.35f;
            float s = Clamp(t, 0.0f, 1.0f);
            s = 1.0f - (1.0f - s) * (1.0f - s);

            glm::mat4 M(1.0f);
            M = glm::translate(M, h.pos);
            glm::vec3 houseSizeAnim = h.scale;
            glm::vec3 up(0,1,0);
            glm::vec3 facing = glm::normalize(h.forward);
            glm::vec3 basisRight = glm::normalize(glm::cross(up, facing));
            glm::mat4 R(1.0f);
            R[0] = glm::vec4(basisRight, 0.0f);
            R[1] = glm::vec4(up, 0.0f);
            R[2] = glm::vec4(facing, 0.0f);
            M = M * R;
            M = glm::scale(M, houseSizeAnim * s);

            ChunkCoord cc = ChunkFromPosXZ(h.pos);
            bool visible = (visibleChunkSet.find(PackChunk(cc.cx, cc.cz)) != visibleChunkSet.end());
            if (visible) {
                float yaw = std::atan2(h.forward.x, h.forward.z);
                animInstances.push_back({glm::vec4(h.pos - renderOrigin, yaw), glm::vec4(houseSizeAnim * s, 0.0f)});
            }

            if (t >= 1.0f) {
                glm::mat4 S(1.0f);
                S = glm::translate(S, h.pos);
                S = S * R;
                S = glm::scale(S, houseSizeAnim);
                state.houseStatic.push_back(S);
                uint64_t ckey = PackChunk(cc.cx, cc.cz);
                state.houseStaticByChunk[ckey].push_back(S);
                BuildingInstance inst;
                inst.asset = h.asset;
                inst.localPos = h.pos;
                inst.yaw = std::atan2(h.forward.x, h.forward.z);
                inst.scale = houseSizeAnim;
                inst.seed = h.seed;
                state.buildingChunks[ckey].instancesByAsset[h.asset].push_back(inst);
                state.dirtyBuildingChunks.insert(ckey);
            } else {
                still.push_back(h);
            }
        }
        state.houseAnim.swap(still);

        // Upload visible chunk houses (static)
        std::vector<RenderHouseBatch> visibleHouseBatches;
        glm::vec3 origin = renderOrigin;
        for (uint64_t key : visibleChunks) {
            auto it = state.buildingChunks.find(key);
            if (it == state.buildingChunks.end()) continue;
            const auto& chunk = it->second;
            for (const auto& assetPair : chunk.instancesByAsset) {
                AssetId assetId = assetPair.first;
                const auto& src = assetPair.second;
                std::vector<HouseInstanceGPU> shifted;
                shifted.reserve(src.size());
                for (const auto& inst : src) {
                    HouseInstanceGPU sInst;
                    sInst.posYaw = glm::vec4(inst.localPos, inst.yaw);
                    sInst.posYaw.x -= origin.x;
                    sInst.posYaw.z -= origin.z;
                    sInst.scaleVar = glm::vec4(inst.scale, 0.0f);
                    shifted.push_back(sInst);
                }
                const MeshGpu& mesh = meshCache.getOrLoad(assetId, assets);
                renderer.updateHouseChunk(key, assetId, mesh, shifted);
                visibleHouseBatches.push_back({key, assetId});
            }
            state.dirtyBuildingChunks.erase(key);
        }

        renderer.updateAnimHouses(animInstances);

        if (state.overlayDirty) {
            RebuildRoadAlignedOverlay(state);
            state.overlayDirty = false;
        }

        // Overlay mesh generation (grid + zones + preview)
        bool showGrid = (mode == Mode::Zone || mode == Mode::Unzone || (mode == Mode::Road && roadTool.drawing));
        std::vector<glm::vec3> buildableVerts;
        std::vector<glm::vec3> zonedResidential;
        std::vector<glm::vec3> zonedCommercial;
        std::vector<glm::vec3> zonedIndustrial;
        std::vector<glm::vec3> zonedOffice;
        std::vector<glm::vec3> waterVerts;
        for (uint64_t key : visibleChunks) {
            auto wit = state.waterChunks.find(key);
            if (showGrid) {
                auto bit = state.overlayBuildableByChunk.find(key);
                if (bit != state.overlayBuildableByChunk.end()) {
                    const auto& src = bit->second;
                    buildableVerts.insert(buildableVerts.end(), src.begin(), src.end());
                }
            }
            auto rit = state.overlayZonedResByChunk.find(key);
            if (rit != state.overlayZonedResByChunk.end()) {
                const auto& src = rit->second;
                zonedResidential.insert(zonedResidential.end(), src.begin(), src.end());
            }
            auto cit = state.overlayZonedComByChunk.find(key);
            if (cit != state.overlayZonedComByChunk.end()) {
                const auto& src = cit->second;
                zonedCommercial.insert(zonedCommercial.end(), src.begin(), src.end());
            }
            auto iit = state.overlayZonedIndByChunk.find(key);
            if (iit != state.overlayZonedIndByChunk.end()) {
                const auto& src = iit->second;
                zonedIndustrial.insert(zonedIndustrial.end(), src.begin(), src.end());
            }
            auto oit = state.overlayZonedOfficeByChunk.find(key);
            if (oit != state.overlayZonedOfficeByChunk.end()) {
                const auto& src = oit->second;
                zonedOffice.insert(zonedOffice.end(), src.begin(), src.end());
            }
            if (wit != state.waterChunks.end()) {
                const WaterChunk& wchunk = wit->second;
                int32_t cx, cz;
                UnpackChunk(key, cx, cz);
                float originX = cx * CHUNK_SIZE_M;
                float originZ = cz * CHUNK_SIZE_M;
                for (int zi = 0; zi < WaterChunk::DIM; ++zi) {
                    for (int xi = 0; xi < WaterChunk::DIM; ++xi) {
                        if (wchunk.get(xi, zi) == 0) continue;
                        AppendWaterCellQuad(waterVerts, originX, originZ, xi, zi);
                    }
                }
            }
        }

        state.zonePreviewVerts.clear();
        if (mode == Mode::Road && roadTool.drawing && roadTool.tempPts.size() >= 2) {
            Road preview;
            preview.pts = roadTool.tempPts;
            preview.rebuildCum();
            BuildRoadPreviewMesh(state, roadTool.tempPts[0], roadTool.tempPts[1]);
            AppendRoadInfluencePreview(state.zonePreviewVerts, preview);
        } else if (mode == Mode::Zone) {
            int rid = zoneTool.dragging ? zoneTool.roadId : zoneTool.hoverRoadId;
            if (rid != -1) {
                int ridx = FindRoadIndexById(state.roads, rid);
                if (ridx >= 0 && state.roads[ridx].pts.size() >= 2) {
                    float a = zoneTool.dragging ? zoneTool.startD : zoneTool.hoverD;
                    float b = zoneTool.dragging ? zoneTool.endD : (zoneTool.hoverD + 40.0f);
                    BuildZonePreviewMesh(state, state.roads[ridx], a, b, zoneTool.sideMask, zoneTool.depth);
                }
            }
        }

        std::vector<glm::vec3> overlayAndPreview;
        overlayAndPreview.reserve(
            buildableVerts.size()
            + zonedResidential.size()
            + zonedCommercial.size()
            + zonedIndustrial.size()
            + zonedOffice.size()
            + state.zonePreviewVerts.size());
        overlayAndPreview.insert(overlayAndPreview.end(), buildableVerts.begin(), buildableVerts.end());
        overlayAndPreview.insert(overlayAndPreview.end(), zonedResidential.begin(), zonedResidential.end());
        overlayAndPreview.insert(overlayAndPreview.end(), zonedCommercial.begin(), zonedCommercial.end());
        overlayAndPreview.insert(overlayAndPreview.end(), zonedIndustrial.begin(), zonedIndustrial.end());
        overlayAndPreview.insert(overlayAndPreview.end(), zonedOffice.begin(), zonedOffice.end());
        overlayAndPreview.insert(overlayAndPreview.end(), state.zonePreviewVerts.begin(), state.zonePreviewVerts.end());

        std::size_t gridCount = buildableVerts.size();
        std::size_t resCount = zonedResidential.size();
        std::size_t comCount = zonedCommercial.size();
        std::size_t indCount = zonedIndustrial.size();
        std::size_t officeCount = zonedOffice.size();
        std::size_t previewCount = state.zonePreviewVerts.size();
        for (auto& v : overlayAndPreview) v -= renderOrigin;
        renderer.updatePreviewMesh(overlayAndPreview);

        for (auto& v : waterVerts) v -= renderOrigin;
        renderer.updateWaterMesh(waterVerts);

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        int houseCount = 0;
        for (const auto& kv : state.buildingChunks) {
            for (const auto& assetPair : kv.second.instancesByAsset) {
                houseCount += (int)assetPair.second.size();
            }
        }
        houseCount += (int)state.houseAnim.size();

        ImGui::Begin("City Painter (Phase 1)");
        const char* modeLabel = "Road (1)";
        if (mode == Mode::Zone) {
            switch (zoneTool.type) {
                case ZoneType::Commercial: modeLabel = "Commercial (3)"; break;
                case ZoneType::Industrial: modeLabel = "Industrial (4)"; break;
                case ZoneType::Office: modeLabel = "Office (5)"; break;
                default: modeLabel = "Residential (2)"; break;
            }
        } else if (mode == Mode::Unzone) {
            modeLabel = "Unzone (6)";
        }
        ImGui::Text("Mode: %s", modeLabel);
        ImGui::Text("Roads: %d", (int)state.roads.size());
        ImGui::Text("Zones: %d", (int)state.zones.size());
        ImGui::Text("Houses: %d", houseCount);
        ImGui::Separator();

        ImGui::Text("Snapping");
        ImGui::Checkbox("Grid snap (G)", &gridSnap);
        ImGui::SliderFloat("Grid size (m)", &gridSize, 1.0f, 20.0f, "%.0f");
        ImGui::Checkbox("Angle snap 15 deg (H)", &angleSnap);
        ImGui::Checkbox("Endpoint snap", &endpointSnap);
        ImGui::SliderFloat("Endpoint radius (m)", &endpointSnapRadius, 2.0f, 30.0f, "%.0f");
        ImGui::SliderFloat("Point pick radius (m)", &roadPointPickRadius, 2.0f, 15.0f, "%.0f");
        ImGui::Separator();

        ImGui::Text("Zoning (depth fixed: %d cells, %.0f m)", ZONE_DEPTH_CELLS, ZONE_DEPTH_M);
        ImGui::Text("Zone type: %s (2-5)", ZoneTypeName(zoneTool.type));
        ImGui::SliderFloat("Zone pick radius (m)", &zoneTool.pickRadius, 4.0f, 30.0f, "%.0f");
        ImGui::Text("Sides (V cycles): %s", (zoneTool.sideMask == 3) ? "Both" : (zoneTool.sideMask == 1) ? "Left" : "Right");
        ImGui::Separator();

        ImGui::Text("Undo/Redo");
        ImGui::Text("Ctrl+Z undo | Ctrl+Y redo | Ctrl+Shift+Z redo");
        ImGui::Separator();

        ImGui::Text("Save/Load (JSON, versioned)");
        ImGui::InputText("File", savePath, sizeof(savePath));
        if (ImGui::Button("Save")) {
            if (SaveToJsonFile(state, assets, savePath)) statusText = "Saved.";
            else statusText = "Save failed.";
        }
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            if (LoadFromJsonFile(state, savePath)) {
                cmds.clear();
                statusText = "Loaded.";
            } else statusText = "Load failed.";
        }
        ImGui::Text("Ctrl+S save | Ctrl+O load");
        ImGui::Separator();

        ImGui::Text("Water Map");
        ImGui::InputText("Water map file", waterMapPath, sizeof(waterMapPath));
        ImGui::SliderFloat("Water threshold", &waterThreshold, 0.0f, 1.0f, "%.2f");
        if (ImGui::Button("Load Water Map")) {
            if (LoadWaterMaskFromImage(state, waterMapPath, waterThreshold)) {
                minimap.dirty = true;
                statusText = "Water map loaded.";
            } else {
                statusText = "Water map load failed.";
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Water")) {
            state.waterChunks.clear();
            state.zonesDirty = true;
            state.housesDirty = true;
            state.overlayDirty = true;
            minimap.dirty = true;
            statusText = "Water cleared.";
        }
        ImGui::Separator();

        ImGui::Text("Lighting");
        ImGui::SliderFloat("Time of day (hours)", &timeOfDayHours, 0.0f, 24.0f, "%.1f");
        ImGui::Separator();

        ImGui::Text("Minimap");
        UpdateMinimapTexture(minimap, state);
        ImVec2 mapSize(240.0f, 240.0f);
        ImGui::Image((ImTextureID)(intptr_t)minimap.texture, mapSize);
        ImVec2 mapMin = ImGui::GetItemRectMin();
        ImVec2 mapMax = ImGui::GetItemRectMax();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        auto mapToScreen = [&](const glm::vec3& pos) {
            float u = (pos.x / MAP_SIDE_M) + 0.5f;
            float v = 0.5f - (pos.z / MAP_SIDE_M);
            u = Clamp(u, 0.0f, 1.0f);
            v = Clamp(v, 0.0f, 1.0f);
            float sx = mapMin.x + u * (mapMax.x - mapMin.x);
            float sy = mapMin.y + v * (mapMax.y - mapMin.y);
            return ImVec2(sx, sy);
        };

        for (const auto& r : state.roads) {
            for (size_t i = 1; i < r.pts.size(); ++i) {
                ImVec2 p0 = mapToScreen(r.pts[i - 1]);
                ImVec2 p1 = mapToScreen(r.pts[i]);
                drawList->AddLine(p0, p1, IM_COL32(220, 220, 220, 160), 1.0f);
            }
        }

        ImVec2 camPos = mapToScreen(cam.target);
        drawList->AddCircleFilled(camPos, 3.0f, IM_COL32(255, 230, 80, 220));

        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            ImVec2 mp = ImGui::GetIO().MousePos;
            float u = (mp.x - mapMin.x) / (mapMax.x - mapMin.x);
            float v = (mp.y - mapMin.y) / (mapMax.y - mapMin.y);
            u = Clamp(u, 0.0f, 1.0f);
            v = Clamp(v, 0.0f, 1.0f);
            cam.target.x = (u - 0.5f) * MAP_SIDE_M;
            cam.target.z = (0.5f - v) * MAP_SIDE_M;
            cam.target.y = 0.0f;
            statusText = "Teleported.";
        }
        ImGui::Separator();

        ImGui::Text("Road editing");
        ImGui::BulletText("Click a road point to select and drag to move");
        ImGui::BulletText("Delete/Backspace deletes selected point (roads keep >= 2 points)");
        ImGui::BulletText("Road drawing: click empty space and hold LMB");
        ImGui::BulletText("To extend: start near an existing road end and draw outward");
        ImGui::BulletText("Zone types: 2 residential, 3 commercial, 4 industrial, 5 office");
        ImGui::BulletText("Unzone (6): click near a road to remove its zones");
        ImGui::BulletText("Zoning won't stack on already-zoned road spans");
        ImGui::Separator();

        ImGui::Text("Status: %s", statusText.c_str());
        ImGui::End();

        ImGui::Render();

        std::vector<RenderMarker> markers;
        if (hasHit && mode == Mode::Road && endpointSnap) {
            glm::vec3 ep; int rid; bool isStart;
            if (SnapToAnyEndpoint(state.roads, mouseHit, endpointSnapRadius, ep, rid, isStart)) {
                markers.push_back({ep - renderOrigin, glm::vec3(1.0f, 0.9f, 0.2f), 1.2f});
            }
        }
        if (hasHit && mode == Mode::Road) {
            markers.push_back({mouseHit - renderOrigin, glm::vec3(0.95f, 0.25f, 0.25f), 0.9f});
        }
        if (hasHit && mode == Mode::Zone && !zoneTool.hoverValid) {
            markers.push_back({mouseHit - renderOrigin, glm::vec3(0.95f, 0.25f, 0.25f), 0.9f});
        }
        if (roadTool.selectedRoadId != -1 && roadTool.selectedPointIndex != -1) {
            int idx = FindRoadIndexById(state.roads, roadTool.selectedRoadId);
            if (idx >= 0 && roadTool.selectedPointIndex < (int)state.roads[idx].pts.size()) {
                glm::vec3 p = state.roads[idx].pts[roadTool.selectedPointIndex] - renderOrigin;
                markers.push_back({p, glm::vec3(0.2f, 0.7f, 1.0f), 1.3f});
            }
        }

        // Shifted road mesh for rendering
        std::vector<RoadVertex> roadRenderVerts;
        roadRenderVerts.reserve(state.roadMeshVerts.size());
        for (const auto& v : state.roadMeshVerts) {
            RoadVertex rv = v;
            rv.pos -= renderOrigin;
            roadRenderVerts.push_back(rv);
        }
        renderer.updateRoadMesh(roadRenderVerts);

        RenderFrame frame;
        frame.viewProj = viewProj;
        frame.viewProjSky = viewProjSky;
        frame.lightViewProj = lightViewProj;
        frame.cameraPos = eye;
        frame.cameraTarget = tgt;
        frame.lighting = lighting;
        frame.roadVertexCount = roadRenderVerts.size();
        frame.waterVertexCount = waterVerts.size();
        frame.gridVertexCount = gridCount;
        frame.zoneResidentialVertexCount = resCount;
        frame.zoneCommercialVertexCount = comCount;
        frame.zoneIndustrialVertexCount = indCount;
        frame.zoneOfficeVertexCount = officeCount;
        frame.previewVertexCount = previewCount;
        frame.visibleHouseBatches = std::move(visibleHouseBatches);
        frame.houseAnimCount = animInstances.size();
        frame.drawRoadPreview = (mode == Mode::Road && roadTool.drawing && !state.zonePreviewVerts.empty());
        frame.zonePreviewValid = zoneTool.dragging ? true : zoneTool.hoverValid;
        frame.zonePreviewType = (uint8_t)zoneTool.type;
        frame.markers = std::move(markers);

        renderer.render(frame);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    if (minimap.texture) {
        glDeleteTextures(1, &minimap.texture);
        minimap.texture = 0;
    }

    renderer.shutdown();
    meshCache.shutdown();

    SDL_GL_DeleteContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
