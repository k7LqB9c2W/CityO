#include "lighting.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace {

float Clamp(float v, float a, float b) {
    return (v < a) ? a : (v > b) ? b : v;
}

float Smoothstep(float a, float b, float x) {
    float t = Clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

} // namespace

LightingParams EvaluateTimeOfDay(float timeHours) {
    float t = std::fmod(timeHours, 24.0f);
    if (t < 0.0f) t += 24.0f;
    float day01 = t / 24.0f;

    float angle = day01 * glm::two_pi<float>() - glm::half_pi<float>();
    float altitude = std::sin(angle);

    float sunUp = Smoothstep(-0.10f, 0.20f, altitude);
    float sunPower = sunUp * Clamp(altitude * 1.25f, 0.0f, 1.0f);

    float azimuth = glm::radians(45.0f);
    glm::vec3 sunDir = glm::normalize(glm::vec3(
        std::cos(azimuth) * std::cos(angle),
        std::sin(angle),
        std::sin(azimuth) * std::cos(angle)
    ));

    glm::vec3 daySun(1.0f, 0.97f, 0.90f);
    glm::vec3 duskSun(1.0f, 0.62f, 0.35f);
    float warm = Smoothstep(-0.10f, 0.05f, altitude) * (1.0f - Smoothstep(0.05f, 0.35f, altitude));
    glm::vec3 sunColor = glm::mix(daySun, duskSun, warm);

    glm::vec3 nightAmb(0.02f, 0.03f, 0.05f);
    glm::vec3 dayAmb(0.45f, 0.50f, 0.55f);
    glm::vec3 ambientColor = glm::mix(nightAmb, dayAmb, sunUp);

    LightingParams out;
    out.sunDir = sunDir;
    out.sunColor = sunColor;
    out.sunIntensity = 1.8f * sunPower;
    out.ambientColor = ambientColor;
    out.ambientIntensity = glm::mix(0.06f, 0.40f, sunUp);
    out.exposure = glm::mix(0.50f, 0.90f, sunUp);
    out.skyExposure = glm::mix(0.60f, 1.15f, sunUp);
    out.skyBrightness = glm::mix(0.0f, 0.85f, sunUp);
    out.shadowStrength = glm::mix(0.65f, 0.90f, sunUp);
    return out;
}

glm::mat4 BuildDirectionalLightMatrix(const glm::vec3& center, float radius, const glm::vec3& sunDir) {
    glm::vec3 lightDir = glm::normalize(-sunDir);
    glm::vec3 up = (std::fabs(lightDir.y) > 0.95f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 lightPos = center - lightDir * radius;
    glm::mat4 view = glm::lookAt(lightPos, center, up);
    glm::mat4 proj = glm::ortho(-radius, radius, -radius, radius, 0.1f, radius * 3.0f);
    return proj * view;
}
