#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "FeatureExtractors.hpp"
#include "Filters.hpp"
#include "Logger.hpp"
#include "Pipeline.hpp"
#include "PointCloudIO.hpp"

namespace fs = std::filesystem;

// ================================================================
// 线程安全的可视化数据容器
// ================================================================
struct VisualizerData {
    struct StageSnapshot {
        std::string title;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointCloud<pcl::Normal>::Ptr normals;
        bool show_normals = false;
    };

    std::mutex mutex;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_original;
    std::vector<StageSnapshot> stage_snapshots;
    bool should_update = false;
    int normal_display_level = 10;

    VisualizerData() {
        cloud_original.reset(new pcl::PointCloud<pcl::PointXYZ>());
    }
};

enum class StepType {
    PassThrough = 0,
    VoxelGrid,
    StatisticalOutlier,
    ShowCloud,
    ShowNormals,
};

struct PipelineStepConfig {
    StepType type = StepType::PassThrough;
};

const char* step_type_to_label(StepType type) {
    switch (type) {
        case StepType::PassThrough:
            return "Filter: PassThrough";
        case StepType::VoxelGrid:
            return "Filter: VoxelGrid";
        case StepType::StatisticalOutlier:
            return "Filter: StatisticalOutlier";
        case StepType::ShowCloud:
            return "Visualize: Show Cloud";
        case StepType::ShowNormals:
            return "Visualize: Compute && Show Normals";
        default:
            return "Unknown";
    }
}

void build_viewer_scene(const pcl::visualization::PCLVisualizer::Ptr& viewer,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& original,
                        const std::vector<VisualizerData::StageSnapshot>& snapshots, int normal_display_level) {
    viewer->setBackgroundColor(0.05, 0.05, 0.05);
    viewer->addCoordinateSystem(0.1);

    const std::size_t total_views = 1 + snapshots.size();
    for (std::size_t idx = 0; idx < total_views; ++idx) {
        int viewport_id = 0;
        const double left = static_cast<double>(idx) / static_cast<double>(total_views);
        const double right = static_cast<double>(idx + 1) / static_cast<double>(total_views);

        viewer->createViewPort(left, 0.0, right, 1.0, viewport_id);
        viewer->setBackgroundColor(0.07 + 0.03 * static_cast<double>(idx % 3),
                                   0.07 + 0.03 * static_cast<double>(idx % 3),
                                   0.07 + 0.03 * static_cast<double>(idx % 3), viewport_id);

        if (idx == 0) {
            viewer->addText("Original", 10, 10, "title_0", viewport_id);
            if (original && !original->empty()) {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_orig(original, 255, 255, 255);
                viewer->addPointCloud<pcl::PointXYZ>(original, color_orig, "cloud_orig", viewport_id);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_orig",
                                                         viewport_id);
            }
            continue;
        }

        const auto& snapshot = snapshots[idx - 1];
        std::string title_id = "title_" + std::to_string(idx);
        std::string cloud_id = "cloud_" + std::to_string(idx);
        std::string normal_id = "normals_" + std::to_string(idx);
        viewer->addText(snapshot.title, 10, 10, title_id, viewport_id);

        if (snapshot.cloud && !snapshot.cloud->empty()) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(snapshot.cloud, 0, 255, 0);
            viewer->addPointCloud<pcl::PointXYZ>(snapshot.cloud, color, cloud_id, viewport_id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id,
                                                     viewport_id);
        }

        if (snapshot.show_normals && snapshot.cloud && snapshot.normals && !snapshot.normals->empty()) {
            viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(
                snapshot.cloud, snapshot.normals, normal_display_level, 0.05, normal_id, viewport_id);
        }
    }
}

// ================================================================
// PCL 可视化线程函数
// ================================================================
void visualizer_thread_func(std::shared_ptr<VisualizerData> vis_data, std::atomic<bool>& keep_running) {
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("PCL Viewer - Dynamic Pipeline"));

    while (keep_running.load() && !viewer->wasStopped()) {
        bool need_rebuild = false;
        pcl::PointCloud<pcl::PointXYZ>::Ptr original_copy(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<VisualizerData::StageSnapshot> snapshots_copy;
        int normal_display_level_copy = 10;

        {
            std::lock_guard<std::mutex> lock(vis_data->mutex);
            if (vis_data->should_update) {
                original_copy = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*vis_data->cloud_original);
                snapshots_copy = vis_data->stage_snapshots;
                normal_display_level_copy = vis_data->normal_display_level;
                need_rebuild = true;
                vis_data->should_update = false;
            }
        }

        if (need_rebuild) {
            viewer.reset(new pcl::visualization::PCLVisualizer("PCL Viewer - Dynamic Pipeline"));
            build_viewer_scene(viewer, original_copy, snapshots_copy, normal_display_level_copy);
        }

        viewer->spinOnce(10);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    viewer->close();
}

// ================================================================
// 错误回调
// ================================================================
static void glfw_error_callback(int error, const char* description) {
    std::cerr << "Glfw Error " << error << ": " << description << '\n';
}

// ================================================================
// 主函数：ImGui 线程
// ================================================================
int main(int argc, char** argv) {
    // 初始化 GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(450, 900, "Point Cloud Pipeline Control", NULL, NULL);
    if (window == NULL) return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // 初始化 ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 创建线程安全的可视化数据容器
    auto vis_data = std::make_shared<VisualizerData>();
    std::atomic<bool> keep_visualizer_running(true);

    // 启动后台可视化线程
    std::thread visualizer_thread(visualizer_thread_func, vis_data, std::ref(keep_visualizer_running));

    // 主线程（ImGui）的状态变量
    std::string current_file = "";
    bool cloud_loaded = false;

    float pass_z_min = -100.0f, pass_z_max = 100.0f;
    float leaf_size = 0.01f;
    int sor_k = 50;
    float sor_std_dev = 1.0f;
    int normal_ksearch = 10;

    std::vector<PipelineStepConfig> pipeline_steps = {
        {StepType::PassThrough},        {StepType::VoxelGrid},   {StepType::ShowCloud},
        {StepType::StatisticalOutlier}, {StepType::ShowNormals},
    };

    auto logger = std::make_shared<ProcessingLog>("logs/processing_gui.log");

    // ================================================================
    // ImGui 主循环（主线程）
    // ================================================================
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Pipeline Control Panel", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        // ========== 文件浏览器 ==========
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "[1] Load Point Cloud");
        ImGui::Text("Available .pcd, .ply, and .bin files in ../data/:");
        ImGui::BeginChild("FileBrowser", ImVec2(0, 120), true);

        std::string data_path = "../data";
        if (fs::exists(data_path) && fs::is_directory(data_path)) {
            for (const auto& entry : fs::directory_iterator(data_path)) {
                if (entry.path().extension() == ".pcd" || entry.path().extension() == ".ply" ||
                    entry.path().extension() == ".bin") {
                    std::string filename = entry.path().filename().string();
                    if (ImGui::Selectable(filename.c_str(), current_file == filename)) {
                        current_file = filename;
                        auto cloud = PointCloudIO::load(entry.path().string());
                        if (cloud && !cloud->empty()) {
                            cloud_loaded = true;
                            {
                                std::lock_guard<std::mutex> lock(vis_data->mutex);
                                vis_data->cloud_original = cloud;
                                vis_data->should_update = true;
                            }
                            std::cout << "[INFO] Loaded " << filename << " (" << cloud->size() << " points)"
                                      << std::endl;
                        }
                    }
                }
            }
        } else {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "ERROR: ../data/ directory not found!");
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Spacing();

        // ========== 参数配置 ==========
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "[2] Parameter Configuration");
        ImGui::DragFloatRange2("PassThrough Z Min/Max", &pass_z_min, &pass_z_max, 0.5f, -200.0f, 200.0f);
        ImGui::SliderFloat("VoxelGrid Leaf Size", &leaf_size, 0.001f, 0.5f, "%.3f m");
        ImGui::SliderInt("StatOutlier Mean K", &sor_k, 1, 200);
        ImGui::SliderFloat("StatOutlier Std Dev", &sor_std_dev, 0.1f, 5.0f);
        ImGui::SliderInt("Normal KSearch", &normal_ksearch, 3, 100);
        ImGui::SliderInt("Normal Display Level", (int*)&vis_data->normal_display_level, 1, 50);

        ImGui::Separator();
        ImGui::Spacing();

        // ========== 自定义 Pipeline 步骤 ==========
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "[3] Custom Pipeline Steps");
        ImGui::TextWrapped("每个 Show Cloud / Show Normals 步骤都会在可视化窗口新增一个视口，展示该步骤时的点云结果。");

        for (std::size_t i = 0; i < pipeline_steps.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Separator();
            ImGui::Text("Step %d", static_cast<int>(i + 1));

            int current_type = static_cast<int>(pipeline_steps[i].type);
            const char* step_labels[] = {
                "Filter: PassThrough",
                "Filter: VoxelGrid",
                "Filter: StatisticalOutlier",
                "Visualize: Show Cloud",
                "Visualize: Compute && Show Normals",
            };
            ImGui::SetNextItemWidth(260.0f);
            ImGui::Combo("Type", &current_type, step_labels, IM_ARRAYSIZE(step_labels));
            pipeline_steps[i].type = static_cast<StepType>(current_type);

            ImGui::SameLine();
            if (ImGui::SmallButton("Up") && i > 0) {
                std::swap(pipeline_steps[i], pipeline_steps[i - 1]);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Down") && i + 1 < pipeline_steps.size()) {
                std::swap(pipeline_steps[i], pipeline_steps[i + 1]);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete") && pipeline_steps.size() > 1) {
                pipeline_steps.erase(pipeline_steps.begin() + static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }

            ImGui::PopID();
        }

        if (ImGui::Button("+ Add Step", ImVec2(-1, 30))) {
            pipeline_steps.push_back({StepType::ShowCloud});
        }

        ImGui::Separator();
        ImGui::Spacing();

        // ========== "计算并刷新"按钮 ==========
        if (ImGui::Button("Compute & Refresh Pipeline", ImVec2(-1, 50)) && cloud_loaded) {
            {
                std::lock_guard<std::mutex> lock(vis_data->mutex);

                auto cloud_to_process = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*vis_data->cloud_original);

                vis_data->stage_snapshots.clear();
                int show_count = 0;

                for (std::size_t i = 0; i < pipeline_steps.size(); ++i) {
                    const StepType type = pipeline_steps[i].type;

                    if (type == StepType::PassThrough) {
                        auto filter = std::make_shared<PassThroughFilter>("z", pass_z_min, pass_z_max);
                        filter->apply(cloud_to_process);
                        continue;
                    }

                    if (type == StepType::VoxelGrid) {
                        auto filter = std::make_shared<VoxelGridFilter>(leaf_size);
                        filter->apply(cloud_to_process);
                        continue;
                    }

                    if (type == StepType::StatisticalOutlier) {
                        auto filter = std::make_shared<StatisticalOutlierFilter>(sor_k, sor_std_dev);
                        filter->apply(cloud_to_process);
                        continue;
                    }

                    VisualizerData::StageSnapshot snapshot;
                    snapshot.cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud_to_process);
                    snapshot.normals.reset(new pcl::PointCloud<pcl::Normal>());
                    snapshot.show_normals = false;

                    if (type == StepType::ShowCloud) {
                        ++show_count;
                        snapshot.title = "View " + std::to_string(show_count) + ": " + step_type_to_label(type);
                    } else if (type == StepType::ShowNormals) {
                        ++show_count;
                        auto normal_extractor = std::make_shared<NormalExtractor>(normal_ksearch, 0.0);
                        auto normals = normal_extractor->extract(cloud_to_process);
                        snapshot.normals = normals;
                        snapshot.show_normals = true;
                        snapshot.title = "View " + std::to_string(show_count) + ": " + step_type_to_label(type);
                    }

                    vis_data->stage_snapshots.push_back(snapshot);
                }

                if (vis_data->stage_snapshots.empty()) {
                    VisualizerData::StageSnapshot final_snapshot;
                    final_snapshot.title = "Final Result";
                    final_snapshot.cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud_to_process);
                    final_snapshot.normals.reset(new pcl::PointCloud<pcl::Normal>());
                    final_snapshot.show_normals = false;
                    vis_data->stage_snapshots.push_back(final_snapshot);
                }

                vis_data->should_update = true;

                logger->log("CustomPipeline", vis_data->cloud_original->size(), cloud_to_process->size());
                std::cout << "[INFO] Custom pipeline executed. Final size: " << cloud_to_process->size() << std::endl;
            }
        }

        if (!cloud_loaded) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Load a point cloud first!");
        }

        ImGui::End();

        // 渲染 ImGui
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // 关闭可视化线程
    keep_visualizer_running = false;
    visualizer_thread.join();

    // 清理 ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}