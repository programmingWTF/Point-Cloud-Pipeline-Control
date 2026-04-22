# Point Cloud Pipeline Control (PointCloudManager)

## 项目简介

本项目是一个交互式点云处理与可视化工具，基于 ImGui + GLFW + OpenGL 的 GUI 与 PCL 的 3D 可视化。用户可以通过可视化流水线组合滤波、下采样、离群点移除、曲率计算与显示步骤，并为每个“显示”步骤生成独立视口以实时比较不同处理阶段的结果。

## 主要功能

- 文件浏览并加载点云：支持 `.pcd`、`.ply`、`.bin`。
- 可组合流水线步骤：PassThrough、VoxelGrid、StatisticalOutlier、计算曲率、显示点云、显示法线。每步独立参数。
- 可视化：原始点云固定视口 + 每个显示步骤单独视口并行显示。
- 实时交互式 GUI：调整参数后一键计算并刷新视图。
- 可扩展：模块化的滤波/特征提取接口，便于添加新算法。

## 目录结构（重要部分）

- `src/` : 源代码（主程序 `main.cpp`、Filters、FeatureExtractors 等）
- `include/` : 头文件
- `extern/imgui` : 内置的 ImGui 源文件（backends 包含 GLFW/OpenGL 后端）
- `data/` : 测试点云数据（默认搜索路径 `../data`）
- `logs/processing_gui.log` : 处理日志

## 依赖

- CMake >= 3.10
- PCL >= 1.8（包含 visualization）
- OpenGL
- GLFW (建议通过 vcpkg 安装)
- Visual Studio (Windows) / GCC (Linux)
- 已包含 ImGui 后端代码在 `extern/imgui`（无需额外安装）

推荐在 Windows 上使用 vcpkg 来安装 PCL/GLFW：

```powershell
# 示例（需先安装 vcpkg 并集成到 CMake）
vcpkg install pcl:x64-windows glfw3:x64-windows
```

## 构建（示例）

在仓库根目录下：

```bash
cmake -S . -B build
cmake --build build --config Release
```

> [!CAUTION]
> 注意：CMakeLists 中使用 `find_package(glfw3 CONFIG REQUIRED)`，在 Windows 下建议使用 vcpkg 来保证依赖被正确发现。

## 运行

生成后直接运行可执行文件（Windows 下为 `PointCloudManager.exe`）。

## 常见问题与调试

- 如果 GUI 中中文显示为问号，检查是否加载到中文字体（程序会尝试寻找系统字体）。
- 点云加载失败：确认 `搜索文件夹` 路径正确，或在 `data/` 中放入测试文件。
- 若在构建时遇到找不到 PCL/GLFW，请确认 vcpkg 已安装对应包并在 CMake 调用中使用 `-DCMAKE_TOOLCHAIN_FILE` 指向 vcpkg 的 toolchain 文件。
