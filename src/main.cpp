#include <SDL.h>
#include <glad/glad.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <nlohmann/json.hpp>

#include "renderer.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <limits>
#include <unordered_set>

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

struct ZoneStrip {
    int id = 0;
    int roadId = 0;
    float d0 = 0.0f;
    float d1 = 0.0f;
    int sideMask = 3; // 1 = left, 2 = right, 3 = both
    float depth = 30.0f;
};

struct HouseAnim {
    glm::vec3 pos;
    float spawnTime;
    glm::vec3 forward;
};

struct AppState {
    int nextRoadId = 1;
    int nextZoneId = 1;

    std::vector<Road> roads;
    std::vector<ZoneStrip> zones;

    bool roadsDirty = true;
    bool housesDirty = true;

    std::vector<glm::vec3> roadMeshVerts;
    std::vector<glm::vec3> zonePreviewVerts;

    std::vector<glm::mat4> houseStatic;
    std::vector<HouseAnim> houseAnim;
};

static bool ZonesOverlap(float a0, float a1, float b0, float b1) {
    float lo = std::max(std::min(a0, a1), std::min(b0, b1));
    float hi = std::min(std::max(a0, a1), std::max(b0, b1));
    return hi >= lo;
}

static bool ZoneOverlapsExisting(const AppState& s, int roadId, float d0, float d1) {
    for (const auto& z : s.zones) {
        if (z.roadId != roadId) continue;
        if (ZonesOverlap(d0, d1, z.d0, z.d1)) return true;
    }
    return false;
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
        s.housesDirty = true;
    }

    void undoIt(AppState& s) override {
        auto it = std::find_if(s.zones.begin(), s.zones.end(), [&](const ZoneStrip& z){ return z.id == zone.id; });
        if (it != s.zones.end()) s.zones.erase(it);
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
        s.housesDirty = true;
    }

    void undoIt(AppState& s) override {
        for (const auto& z : removed) s.zones.push_back(z);
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
    const float roadWidth = 10.0f;
    const float y = 0.03f;

    for (const auto& r : s.roads) {
        if (r.pts.size() < 2) continue;
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

            s.roadMeshVerts.push_back(aL);
            s.roadMeshVerts.push_back(aR);
            s.roadMeshVerts.push_back(bR);

            s.roadMeshVerts.push_back(aL);
            s.roadMeshVerts.push_back(bR);
            s.roadMeshVerts.push_back(bL);
        }
    }
}

static void AppendZoneMesh(
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

    const float roadHalf = 5.0f;
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

static void BuildZonePreviewMesh(
    AppState& s,
    const Road& r,
    float d0,
    float d1,
    int sideMask,
    float depth)
{
    s.zonePreviewVerts.clear();
    AppendZoneMesh(s.zonePreviewVerts, r, d0, d1, sideMask, depth);
}

static void BuildRoadPreviewMesh(AppState& s, const glm::vec3& a, const glm::vec3& b) {
    const float roadWidth = 10.0f;
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

static void RebuildHousesFromZones(AppState& s, bool animate, float nowSec) {
    s.houseStatic.clear();
    s.houseAnim.clear();

    const glm::vec3 houseSize = glm::vec3(8.0f, 6.0f, 12.0f); // width, height, depth
    const float roadHalf = 5.0f;
    const float desiredClear = 10.0f;          // minimum gap from road edge to house front
    const float setback = roadHalf + desiredClear + houseSize.z * 0.5f; // centerline to house center
    const float spacing = 15.0f;               // along the road
    const float rowSpacing = 14.0f;            // away from road

    for (const auto& z : s.zones) {
        int ridx = FindRoadIndexById(s.roads, z.roadId);
        if (ridx < 0) continue;
        const Road& r = s.roads[ridx];
        if (r.pts.size() < 2) continue;

        float a = std::min(z.d0, z.d1);
        float b = std::max(z.d0, z.d1);

        int rows = std::max(1, (int)std::floor(z.depth / rowSpacing));

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

        for (float d = a; d <= b; d += spacing) {
            glm::vec3 tan;
            glm::vec3 p = r.pointAt(d, tan);
            glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0,1,0), tan));

            auto doSide = [&](int side) {
                for (int row = 0; row < rows; row++) {
                    glm::vec3 pos = p + right * float(side) * (setback + row * rowSpacing);
                    pos.y = houseSize.y * 0.5f;

                    float distSq = minCenterlineClearSq(pos);
                    float clearFromEdge = std::sqrt(distSq) - roadHalf; // distance from nearest road edge
                    if (clearFromEdge < desiredClear) continue; // too close to any road (intersections)
                    if (isOccupied(pos)) continue; // avoid double builds/overlap

                    glm::vec3 up(0,1,0);
                    glm::vec3 facing = glm::normalize(-float(side) * right); // face toward road
                    glm::vec3 basisRight = glm::normalize(glm::cross(up, facing));
                    glm::mat4 R(1.0f);
                    R[0] = glm::vec4(basisRight, 0.0f);
                    R[1] = glm::vec4(up, 0.0f);
                    R[2] = glm::vec4(facing, 0.0f);

                    if (animate) {
                        uint32_t hx = (uint32_t)std::llround(pos.x * 10.0);
                        uint32_t hz = (uint32_t)std::llround(pos.z * 10.0);
                        uint32_t h = Hash32(hx ^ (hz * 1664525U) ^ (uint32_t)z.id);
                        float jitter = (h % 120) / 1000.0f; // 0..0.119 sec
                        s.houseAnim.push_back({pos, nowSec + jitter, facing});
                    } else {
                        glm::mat4 M(1.0f);
                        M = glm::translate(M, pos);
                        M = M * R;
                        M = glm::scale(M, houseSize);
                        s.houseStatic.push_back(M);
                    }
                    markOccupied(pos);
                }
            };

            if (z.sideMask & 1) doSide(-1);
            if (z.sideMask & 2) doSide(+1);
        }
    }
}

static bool SaveToJsonFile(const AppState& s, const std::string& path) {
    json j;
    j["version"] = 1;
    j["nextRoadId"] = s.nextRoadId;
    j["nextZoneId"] = s.nextZoneId;

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
        jz["depth"] = z.depth;
        j["zones"].push_back(jz);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << j.dump(2);
    return true;
}

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
        z.depth = jz.value("depth", 30.0f);
        s.zones.push_back(z);
    }

    s.roadsDirty = true;
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
    float depth = 30.0f;
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
    std::string statusText;

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

        float aspect = (winH > 0) ? (float)winW / (float)winH : 1.0f;
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500000.0f);
        glm::mat4 view = cam.view();
        glm::mat4 viewProj = proj * view;

        glm::vec3 mouseHit;
        bool hasHit = ScreenToGroundHit(mx, my, winW, winH, view, proj, mouseHit);

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
                if (k == SDLK_2) { mode = Mode::Zone; statusText = "Zone mode."; }
                if (k == SDLK_3) { mode = Mode::Unzone; statusText = "Unzone mode."; }

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
                    if (SaveToJsonFile(state, savePath)) statusText = "Saved.";
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
                    // If clicking near a road point: start moving interior points, but endpoints start a new road.
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
                            // Start a new road anchored to this endpoint; do not move existing geometry.
                            roadTool.selectedRoadId = -1;
                            roadTool.selectedPointIndex = -1;
                            startRoadDraw(state.roads[idx].pts[pi]);
                            statusText = "Extending from endpoint (new road).";
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
                        z.depth = zoneTool.depth;

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

        // Rebuild road mesh if dirty
        if (state.roadsDirty) {
            for (auto& r : state.roads) {
                if (r.cumLen.size() != r.pts.size()) r.rebuildCum();
            }
            RebuildAllRoadMesh(state);
            renderer.updateRoadMesh(state.roadMeshVerts);
            state.roadsDirty = false;
        }

        // Rebuild houses if zones changed
        if (state.housesDirty) {
            bool animate = true; // animate after zone/road edits for now
            RebuildHousesFromZones(state, animate, nowSec);
            state.housesDirty = false;
        }

        // House animation step (move finished anim houses into static instances)
        std::vector<glm::mat4> animModelsTmp;
        animModelsTmp.reserve(state.houseAnim.size());

        std::vector<HouseAnim> still;
        still.reserve(state.houseAnim.size());

        for (const auto& h : state.houseAnim) {
            float t = (nowSec - h.spawnTime) / 0.35f;
            float s = Clamp(t, 0.0f, 1.0f);
            s = 1.0f - (1.0f - s) * (1.0f - s);

            glm::mat4 M(1.0f);
            M = glm::translate(M, h.pos);
            // same houseSize as static, scaled by anim factor
            const glm::vec3 houseSizeAnim(8.0f, 6.0f, 12.0f);
            glm::vec3 up(0,1,0);
            glm::vec3 facing = glm::normalize(h.forward);
            glm::vec3 basisRight = glm::normalize(glm::cross(up, facing));
            glm::mat4 R(1.0f);
            R[0] = glm::vec4(basisRight, 0.0f);
            R[1] = glm::vec4(up, 0.0f);
            R[2] = glm::vec4(facing, 0.0f);
            M = M * R;
            M = glm::scale(M, houseSizeAnim * s);
            animModelsTmp.push_back(M);

            if (t >= 1.0f) {
                glm::mat4 S(1.0f);
                S = glm::translate(S, h.pos);
                S = S * R;
                S = glm::scale(S, houseSizeAnim);
                state.houseStatic.push_back(S);
            } else {
                still.push_back(h);
            }
        }
        state.houseAnim.swap(still);

        renderer.updateHouseInstances(state.houseStatic, animModelsTmp);

        // Overlay mesh generation (existing zones + preview)
        zoneTool.overlayVerts.clear();
        for (const auto& z : state.zones) {
            int ridx = FindRoadIndexById(state.roads, z.roadId);
            if (ridx >= 0 && state.roads[ridx].pts.size() >= 2) {
                AppendZoneMesh(zoneTool.overlayVerts, state.roads[ridx], z.d0, z.d1, z.sideMask, z.depth);
            }
        }

        state.zonePreviewVerts.clear();
        if (mode == Mode::Road && roadTool.drawing && roadTool.tempPts.size() >= 2) {
            BuildRoadPreviewMesh(state, roadTool.tempPts[0], roadTool.tempPts[1]);
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
        std::vector<glm::vec3> overlayAndPreview = zoneTool.overlayVerts;
        overlayAndPreview.insert(overlayAndPreview.end(), state.zonePreviewVerts.begin(), state.zonePreviewVerts.end());
        renderer.updatePreviewMesh(overlayAndPreview);

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("City Painter (Phase 1)");
        const char* modeLabel = (mode == Mode::Road) ? "Road (1)" : (mode == Mode::Zone) ? "Zone (2)" : "Unzone (3)";
        ImGui::Text("Mode: %s", modeLabel);
        ImGui::Text("Roads: %d", (int)state.roads.size());
        ImGui::Text("Zones: %d", (int)state.zones.size());
        ImGui::Text("Houses: %d", (int)(state.houseStatic.size() + state.houseAnim.size()));
        ImGui::Separator();

        ImGui::Text("Snapping");
        ImGui::Checkbox("Grid snap (G)", &gridSnap);
        ImGui::SliderFloat("Grid size (m)", &gridSize, 1.0f, 20.0f, "%.0f");
        ImGui::Checkbox("Angle snap 15 deg (H)", &angleSnap);
        ImGui::Checkbox("Endpoint snap", &endpointSnap);
        ImGui::SliderFloat("Endpoint radius (m)", &endpointSnapRadius, 2.0f, 30.0f, "%.0f");
        ImGui::SliderFloat("Point pick radius (m)", &roadPointPickRadius, 2.0f, 15.0f, "%.0f");
        ImGui::Separator();

        ImGui::Text("Zoning");
        ImGui::SliderFloat("Zone depth (m)", &zoneTool.depth, 10.0f, 120.0f, "%.0f");
        ImGui::SliderFloat("Zone pick radius (m)", &zoneTool.pickRadius, 4.0f, 30.0f, "%.0f");
        ImGui::Text("Sides (V cycles): %s", (zoneTool.sideMask == 3) ? "Both" : (zoneTool.sideMask == 1) ? "Left" : "Right");
        ImGui::Separator();

        ImGui::Text("Undo/Redo");
        ImGui::Text("Ctrl+Z undo | Ctrl+Y redo | Ctrl+Shift+Z redo");
        ImGui::Separator();

        ImGui::Text("Save/Load (JSON, versioned)");
        ImGui::InputText("File", savePath, sizeof(savePath));
        if (ImGui::Button("Save")) {
            if (SaveToJsonFile(state, savePath)) statusText = "Saved.";
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

        ImGui::Text("Road editing");
        ImGui::BulletText("Click a road point to select and drag to move");
        ImGui::BulletText("Delete/Backspace deletes selected point (roads keep >= 2 points)");
        ImGui::BulletText("Road drawing: click empty space and hold LMB");
        ImGui::BulletText("To extend: start near an existing road end and draw outward");
        ImGui::BulletText("Unzone (3): click near a road to remove its zones");
        ImGui::BulletText("Zoning won't stack on already-zoned road spans");
        ImGui::Separator();

        ImGui::Text("Status: %s", statusText.c_str());
        ImGui::End();

        ImGui::Render();

        std::vector<RenderMarker> markers;
        if (hasHit && mode == Mode::Road && endpointSnap) {
            glm::vec3 ep; int rid; bool isStart;
            if (SnapToAnyEndpoint(state.roads, mouseHit, endpointSnapRadius, ep, rid, isStart)) {
                markers.push_back({ep, glm::vec3(1.0f, 0.9f, 0.2f), 1.2f});
            }
        }
        if (hasHit && mode == Mode::Zone && !zoneTool.hoverValid) {
            markers.push_back({mouseHit, glm::vec3(0.95f, 0.25f, 0.25f), 0.9f});
        }
        if (roadTool.selectedRoadId != -1 && roadTool.selectedPointIndex != -1) {
            int idx = FindRoadIndexById(state.roads, roadTool.selectedRoadId);
            if (idx >= 0 && roadTool.selectedPointIndex < (int)state.roads[idx].pts.size()) {
                markers.push_back({state.roads[idx].pts[roadTool.selectedPointIndex], glm::vec3(0.2f, 0.7f, 1.0f), 1.3f});
            }
        }

        RenderFrame frame;
        frame.viewProj = viewProj;
        frame.roadVertexCount = state.roadMeshVerts.size();
        frame.overlayVertexCount = zoneTool.overlayVerts.size();
        frame.previewVertexCount = state.zonePreviewVerts.size();
        frame.houseStaticCount = state.houseStatic.size();
        frame.houseAnimCount = animModelsTmp.size();
        frame.drawRoadPreview = (mode == Mode::Road && roadTool.drawing && !state.zonePreviewVerts.empty());
        frame.zonePreviewValid = zoneTool.dragging ? true : zoneTool.hoverValid;
        frame.markers = std::move(markers);

        renderer.render(frame);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    renderer.shutdown();

    SDL_GL_DeleteContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
