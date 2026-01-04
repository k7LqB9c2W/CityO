cmake_minimum_required(VERSION 3.20)
project(CityPainterProto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CityPainterProto
  src/main.cpp
  src/renderer.cpp
  src/asset_catalog.cpp
  src/mesh_cache.cpp
  src/image_loader.cpp

  external/imgui/imgui.cpp
  external/imgui/imgui_draw.cpp
  external/imgui/imgui_tables.cpp
  external/imgui/imgui_widgets.cpp

  external/imgui/backends/imgui_impl_sdl2.cpp
  external/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(CityPainterProto PRIVATE
  external/imgui
  external/imgui/backends
  external/cgltf
)

target_compile_definitions(CityPainterProto PRIVATE
  SDL_MAIN_HANDLED
  IMGUI_IMPL_OPENGL_LOADER_GLAD
)

find_package(SDL2 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(CityPainterProto PRIVATE
  SDL2::SDL2
  glad::glad
  glm::glm
  nlohmann_json::nlohmann_json
  opengl32
  windowscodecs
)


# Copy SDL2.dll next to the exe automatically when using vcpkg toolchain
if (DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  add_custom_command(TARGET CityPainterProto POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/SDL2.dll"
      "$<TARGET_FILE_DIR:CityPainterProto>/SDL2.dll"
    VERBATIM
  )
endif()
