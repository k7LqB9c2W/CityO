#pragma once

#include <cstdint>
#include <vector>

bool LoadImageRGBA(const char* path, std::vector<uint8_t>& outPixels, int& outW, int& outH);
