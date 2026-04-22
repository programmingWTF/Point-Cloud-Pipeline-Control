cmake_minimum_required(VERSION 3.10)
project(PointCloudManager)

# 1. 寻找依赖库: PCL, OpenGL, GLFW
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters visualization features kdtree search)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED) # 假设用户使用 vcpkg 或 CMake Config 安装了 glfw3

# 2. 找到本项目的所有的源文件
file(GLOB SOURCES "src/*.cpp")

# 3. 添加 ImGui 源文件
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui)
file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

# 4. 生成可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${IMGUI_SOURCES})

# 5. 配置目标属性
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${PCL_INCLUDE_DIRS}
)

# 6. 链接库文件
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PCL_LIBRARIES}
    OpenGL::GL
    glfw
)