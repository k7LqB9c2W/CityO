#pragma once

#include <glm/glm.hpp>

struct LightingParams {
    glm::vec3 sunDir{0.0f, 1.0f, 0.0f};
    glm::vec3 sunColor{1.0f, 0.97f, 0.90f};
    float sunIntensity = 1.0f;
    glm::vec3 ambientColor{0.45f, 0.50f, 0.55f};
    float ambientIntensity = 0.6f;
    float exposure = 1.0f;
    float skyBrightness = 1.0f;
    float shadowStrength = 0.85f;
};

LightingParams EvaluateTimeOfDay(float timeHours);
glm::mat4 BuildDirectionalLightMatrix(const glm::vec3& center, float radius, const glm::vec3& sunDir);
