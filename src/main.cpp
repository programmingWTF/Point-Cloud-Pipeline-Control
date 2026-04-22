#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <atomic>
#include <chrono>
#include <cstdio>
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

struct DisplayConfig {
    int color_r = 0;
    int color_g = 255;
    int color_b = 0;
    int point_size = 2;

    bool show_normals = false;
    int normal_level = 10;
    float normal_scale = 0.05F;

    bool show_coordinate_axis = false;
    float axis_scale = 0.10F;

    float background_r = 0.10F;
    float background_g = 0.10F;
    float background_b = 0.10F;
};

// ================================================================
// 线程安全的可视化数据容器
// ================================================================
struct VisualizerData {
    struct StageSnapshot {
        std::string title;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointCloud<pcl::Normal>::Ptr normals;
        DisplayConfig display;
    };

    std::mutex mutex;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_original;
    DisplayConfig original_display;

    std::vector<StageSnapshot> stage_snapshots;
    bool should_update = false;

    VisualizerData() {
        cloud_original.reset(new pcl::PointCloud<pcl::PointXYZ>());
        original_display.color_r = 255;
        original_display.color_g = 255;
        original_display.color_b = 255;
        original_display.point_size = 2;
        original_display.show_coordinate_axis = true;
        original_display.axis_scale = 0.12F;
        original_display.background_r = 0.07F;
        original_display.background_g = 0.07F;
        original_display.background_b = 0.07F;
    }
};

enum class StepType {
    PassThrough = 0,
    VoxelGrid,
    StatisticalOutlier,
    ComputeCurvature,
    ShowCloud,
    ShowNormals,
};

struct PipelineStepConfig {
    StepType type = StepType::PassThrough;

    // PassThrough
    int pass_axis_idx = 2;  // 0:x, 1:y, 2:z
    float pass_min = -100.0f;
    float pass_max = 100.0f;

    // VoxelGrid
    float voxel_leaf_size = 0.01f;

    // StatisticalOutlier
    int sor_k = 50;
    float sor_std_dev = 1.0f;

    // Curvature
    int curvature_ksearch = 10;
    float curvature_radius = 0.0f;

    // Normals for display
    int normal_ksearch = 10;
    float normal_radius = 0.0f;

    // Display
    DisplayConfig display;
};

const char* step_type_to_label(StepType type) {
    switch (type) {
        case StepType::PassThrough:
            return "Filter: PassThrough";
        case StepType::VoxelGrid:
            return "Filter: VoxelGrid";
        case StepType::StatisticalOutlier:
            return "Filter: StatisticalOutlier";
        case StepType::ComputeCurvature:
            return "Feature: Compute Curvature";
        case StepType::ShowCloud:
            return "Visualize: Show Cloud";
        case StepType::ShowNormals:
            return "Visualize: Show Normals";
        default:
            return "Unknown";
    }
}

const char* axis_idx_to_field(int axis_idx) {
    switch (axis_idx) {
        case 0:
            return "x";
        case 1:
            return "y";
        default:
            return "z";
    }
}

bool is_display_step(StepType type) {
    return type == StepType::ShowCloud || type == StepType::ShowNormals;
}

bool setup_imgui_font(float font_pixel_size, std::string& loaded_font_path) {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    const ImWchar* glyph_ranges = io.Fonts->GetGlyphRangesChineseFull();
    const char* font_candidates[] = {
        "../fonts/MSYH.TTC",
        "C:/Windows/Fonts/MSYH.ttc",
        "C:/Windows/Fonts/MSYH.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    };

    for (const char* font_path : font_candidates) {
        if (!fs::exists(font_path)) {
            continue;
        }
        ImFont* font = io.Fonts->AddFontFromFileTTF(font_path, font_pixel_size, nullptr, glyph_ranges);
        if (font != nullptr) {
            loaded_font_path = font_path;
            io.FontDefault = font;
            io.FontGlobalScale = 1.0f;
            return true;
        }
    }

    io.Fonts->AddFontDefault();
    io.FontGlobalScale = 1.30f;
    loaded_font_path = "ImGui Default Font";
    return false;
}

PipelineStepConfig make_default_step(StepType type) {
    PipelineStepConfig step;
    step.type = type;

    step.display.color_r = 0;
    step.display.color_g = 255;
    step.display.color_b = 0;
    step.display.point_size = 2;
    step.display.show_normals = (type == StepType::ShowNormals);
    step.display.normal_level = 10;
    step.display.normal_scale = 0.05F;
    step.display.show_coordinate_axis = false;
    step.display.axis_scale = 0.10F;
    step.display.background_r = 0.10F;
    step.display.background_g = 0.10F;
    step.display.background_b = 0.10F;

    return step;
}

void render_display_config_ui(DisplayConfig& config, bool allow_normals_edit) {
    ImGui::SliderInt("Point Size", &config.point_size, 1, 8);
    ImGui::SliderInt("点云颜色 R", &config.color_r, 0, 255);
    ImGui::SliderInt("点云颜色 G", &config.color_g, 0, 255);
    ImGui::SliderInt("点云颜色 B", &config.color_b, 0, 255);

    ImGui::Checkbox("显示坐标轴", &config.show_coordinate_axis);
    if (config.show_coordinate_axis) {
        ImGui::SliderFloat("坐标轴刻度", &config.axis_scale, 0.02F, 1.0F, "%.2F");
    }

    ImGui::SliderFloat("背景颜色 R", &config.background_r, 0.0F, 1.0F, "%.2F");
    ImGui::SliderFloat("背景颜色 G", &config.background_g, 0.0F, 1.0F, "%.2F");
    ImGui::SliderFloat("背景颜色 B", &config.background_b, 0.0F, 1.0F, "%.2F");
}

void build_viewer_scene(const pcl::visualization::PCLVisualizer::Ptr& viewer,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& original, const DisplayConfig& original_display,
                        const std::vector<VisualizerData::StageSnapshot>& snapshots) {
    const std::size_t total_views = 1 + snapshots.size();
    for (std::size_t idx = 0; idx < total_views; ++idx) {
        int viewport_id = 0;
        const double left = static_cast<double>(idx) / static_cast<double>(total_views);
        const double right = static_cast<double>(idx + 1) / static_cast<double>(total_views);

        viewer->createViewPort(left, 0.0, right, 1.0, viewport_id);

        DisplayConfig display;
        std::string title = "Original";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_show = original;
        pcl::PointCloud<pcl::Normal>::Ptr normals_to_show(new pcl::PointCloud<pcl::Normal>());

        if (idx == 0) {
            display = original_display;
        } else {
            const auto& snapshot = snapshots[idx - 1];
            display = snapshot.display;
            title = snapshot.title;
            cloud_to_show = snapshot.cloud;
            normals_to_show = snapshot.normals;
        }

        viewer->setBackgroundColor(display.background_r, display.background_g, display.background_b, viewport_id);

        std::string title_id = "title_" + std::to_string(idx);
        std::string cloud_id = "cloud_" + std::to_string(idx);
        std::string normal_id = "normals_" + std::to_string(idx);
        std::string axis_id = "axis_" + std::to_string(idx);

        viewer->addText(title, 10, 10, title_id, viewport_id);

        if (display.show_coordinate_axis) {
            viewer->addCoordinateSystem(display.axis_scale, axis_id, viewport_id);
        }

        if (cloud_to_show && !cloud_to_show->empty()) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud_to_show, display.color_r,
                                                                                  display.color_g, display.color_b);
            viewer->addPointCloud<pcl::PointXYZ>(cloud_to_show, color, cloud_id, viewport_id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     static_cast<double>(display.point_size), cloud_id, viewport_id);
        }

        if (display.show_normals && cloud_to_show && normals_to_show && !normals_to_show->empty()) {
            viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(
                cloud_to_show, normals_to_show, display.normal_level, display.normal_scale, normal_id, viewport_id);
        }
    }
}

// ================================================================
// PCL 可视化线程函数
// ================================================================
void visualizer_thread_func(std::shared_ptr<VisualizerData> vis_data, std::atomic<bool>& keep_running) {
    pcl::visualization::PCLVisualizer::Ptr viewer;

    while (keep_running.load()) {
        bool viewer_stopped = (viewer && viewer->wasStopped());
        bool need_rebuild = false;

        pcl::PointCloud<pcl::PointXYZ>::Ptr original_copy(new pcl::PointCloud<pcl::PointXYZ>());
        DisplayConfig original_display_copy;
        std::vector<VisualizerData::StageSnapshot> snapshots_copy;

        {
            std::lock_guard<std::mutex> lock(vis_data->mutex);
            if (vis_data->should_update) {
                original_copy = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*vis_data->cloud_original);
                original_display_copy = vis_data->original_display;
                snapshots_copy = vis_data->stage_snapshots;
                need_rebuild = true;
                vis_data->should_update = false;
            }
        }

        // 如果窗口被关闭或需要更新数据，重新创建窗口
        if (viewer_stopped || (need_rebuild && (!original_copy->empty() || !snapshots_copy.empty()))) {
            if (viewer) {
                viewer->close();
                viewer.reset();
            }
            viewer.reset(new pcl::visualization::PCLVisualizer("PCL Viewer - 点云3D视图"));
            build_viewer_scene(viewer, original_copy, original_display_copy, snapshots_copy);
        }

        if (viewer && !viewer->wasStopped()) {
            viewer->spinOnce(10);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (viewer) {
        viewer->close();
    }
}

// ================================================================
// 错误回调
// ================================================================
static void glfw_error_callback(int error, const char* description) {
    std::cerr << "Glfw Error " << error << ": " << description << '\n';
}

// 全局变量用于处理窗口关闭确认
static bool g_request_exit = false;

static void glfw_window_close_callback(GLFWwindow* window) {
    // 阻止默认关闭，设置标志让 ImGui 显示确认对话框
    glfwSetWindowShouldClose(window, GLFW_FALSE);
    g_request_exit = true;
}

// ================================================================
// 主函数：ImGui 线程
// ================================================================
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(560, 980, "Point Cloud Pipeline Control", NULL, NULL);
    if (window == NULL) return 1;

    glfwSetWindowCloseCallback(window, glfw_window_close_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui::GetStyle().ScaleAllSizes(1.15f);
    std::string font_loaded_path;
    const bool has_chinese_font = setup_imgui_font(24.0f, font_loaded_path);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    if (has_chinese_font) {
        std::cout << "[INFO] ImGui font loaded: " << font_loaded_path << "\n";
    } else {
        std::cout << "[WARN] No Chinese font found in system candidates. GUI text may show '?' for Chinese."
                  << "\n";
    }

    auto vis_data = std::make_shared<VisualizerData>();
    std::atomic<bool> keep_visualizer_running(true);
    std::thread visualizer_thread(visualizer_thread_func, vis_data, std::ref(keep_visualizer_running));

    bool cloud_loaded = false;
    std::string current_file_name;
    std::string current_file_path;

    char search_dir[512] = "../data";

    std::vector<PipelineStepConfig> pipeline_steps = {
        make_default_step(StepType::PassThrough),      make_default_step(StepType::VoxelGrid),
        make_default_step(StepType::ComputeCurvature), make_default_step(StepType::ShowCloud),
        make_default_step(StepType::ShowNormals),
    };

    int add_step_type_idx = static_cast<int>(StepType::ShowCloud);

    auto logger = std::make_shared<ProcessingLog>("logs/processing_gui.log");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Pipeline Control Panel", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        // 1) 文件选择区
        ImGui::SeparatorText("文件选择区");
        ImGui::InputText("搜索文件夹", search_dir, IM_ARRAYSIZE(search_dir));
        ImGui::TextWrapped("支持 .pcd / .ply / .bin；可直接修改上方路径来搜索。当前：%s", search_dir);

        ImGui::BeginChild("FileBrowser", ImVec2(0, 160), true);
        if (fs::exists(search_dir) && fs::is_directory(search_dir)) {
            for (const auto& entry : fs::directory_iterator(search_dir)) {
                const auto ext = entry.path().extension().string();
                if (ext != ".pcd" && ext != ".ply" && ext != ".bin") {
                    continue;
                }

                const std::string filename = entry.path().filename().string();
                const bool selected = (current_file_path == entry.path().string());
                if (ImGui::Selectable(filename.c_str(), selected)) {
                    current_file_name = filename;
                    current_file_path = entry.path().string();
                }
            }
        } else {
            ImGui::TextColored(ImVec4(1, 0.2f, 0.2f, 1), "路径不可用或不是文件夹。请检查搜索文件夹路径。");
        }
        ImGui::EndChild();

        if (ImGui::Button("加载当前选择文件", ImVec2(-1, 34))) {
            if (!current_file_path.empty()) {
                auto cloud = PointCloudIO::load(current_file_path);
                if (cloud && !cloud->empty()) {
                    cloud_loaded = true;
                    {
                        std::lock_guard<std::mutex> lock(vis_data->mutex);
                        vis_data->cloud_original = cloud;
                        vis_data->stage_snapshots.clear();
                        vis_data->should_update = true;
                    }
                    std::cout << "[INFO] Loaded " << current_file_name << " (" << cloud->size() << " points)"
                              << "\n";
                } else {
                    std::cout << "[WARN] Failed to load selected file: " << current_file_path << "\n";
                }
            }
        }

        if (!current_file_name.empty()) {
            ImGui::Text("已选择：%s", current_file_name.c_str());
        }

        // 2) 流水线自定义区域
        ImGui::SeparatorText("流水线自定义区域");
        ImGui::TextWrapped(
            "这里可以组合滤波器、曲率计算和显示步骤。"
            " 只有显示步骤会新增可视化视口。\n"
            "步骤支持：PassThrough、VoxelGrid、StatisticalOutlier、计算曲率、显示点云、显示法线。");

        const char* step_labels[] = {
            "滤波：PassThrough", "滤波：VoxelGrid",  "滤波：StatisticalOutlier",
            "特征：计算曲率",    "可视化：显示点云", "可视化：显示法线",
        };

        for (std::size_t i = 0; i < pipeline_steps.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Separator();
            ImGui::Text("Step %d", static_cast<int>(i + 1));

            int current_type = static_cast<int>(pipeline_steps[i].type);
            ImGui::SetNextItemWidth(300.0f);
            if (ImGui::Combo("Type", &current_type, step_labels, IM_ARRAYSIZE(step_labels))) {
                pipeline_steps[i] = make_default_step(static_cast<StepType>(current_type));
            }

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

            const StepType type = pipeline_steps[i].type;

            if (type == StepType::PassThrough) {
                const char* axis_labels[] = {"x", "y", "z"};
                ImGui::Combo("Axis", &pipeline_steps[i].pass_axis_idx, axis_labels, IM_ARRAYSIZE(axis_labels));
                ImGui::DragFloatRange2("Min/Max", &pipeline_steps[i].pass_min, &pipeline_steps[i].pass_max, 0.2f,
                                       -200.0f, 200.0f);
            } else if (type == StepType::VoxelGrid) {
                ImGui::SliderFloat("Leaf Size", &pipeline_steps[i].voxel_leaf_size, 0.001f, 0.5f, "%.3f m");
            } else if (type == StepType::StatisticalOutlier) {
                ImGui::SliderInt("Mean K", &pipeline_steps[i].sor_k, 1, 200);
                ImGui::SliderFloat("Std Dev", &pipeline_steps[i].sor_std_dev, 0.1f, 5.0f, "%.2f");
            } else if (type == StepType::ComputeCurvature) {
                ImGui::SliderInt("Curvature KSearch", &pipeline_steps[i].curvature_ksearch, 3, 100);
                ImGui::SliderFloat("Curvature Radius", &pipeline_steps[i].curvature_radius, 0.0f, 1.0f, "%.3f");
                ImGui::TextWrapped("Radius > 0 时使用半径搜索，否则使用 KSearch。");
            } else if (type == StepType::ShowCloud || type == StepType::ShowNormals) {
                ImGui::SliderInt("Normal KSearch", &pipeline_steps[i].normal_ksearch, 3, 100);
                ImGui::SliderFloat("Normal Radius", &pipeline_steps[i].normal_radius, 0.0f, 1.0f, "%.3f");
                if (type == StepType::ShowNormals) {
                    pipeline_steps[i].display.show_normals = true;
                }
                render_display_config_ui(pipeline_steps[i].display, type != StepType::ShowNormals);
            }

            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::SetNextItemWidth(300.0f);
        ImGui::Combo("新增步骤类型", &add_step_type_idx, step_labels, IM_ARRAYSIZE(step_labels));
        if (ImGui::Button("+ Add Step", ImVec2(-1, 30))) {
            pipeline_steps.push_back(make_default_step(static_cast<StepType>(add_step_type_idx)));
        }

        // 3) 可视化部分
        ImGui::SeparatorText("原始点云的可视化部分");
        ImGui::TextWrapped(
            "原始点云固定有一个视口。流水线里每个显示步骤会新增一个视口，并按该步骤中的显示参数渲染"
            "（颜色、点大小、是否显示法线、法线密度/长度、是否显示坐标轴、背景色）。");

        if (ImGui::CollapsingHeader("原始点云显示参数", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_display_config_ui(vis_data->original_display, false);
        }

        int display_step_count = 0;
        for (const auto& step : pipeline_steps) {
            if (is_display_step(step.type)) {
                ++display_step_count;
            }
        }
        ImGui::Text("预计视口数量：%d（1 个原始 + %d 个显示步骤）", 1 + display_step_count, display_step_count);

        if (ImGui::Button("计算并刷新视图", ImVec2(-1, 48)) && cloud_loaded) {
            std::lock_guard<std::mutex> lock(vis_data->mutex);

            auto cloud_to_process = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*vis_data->cloud_original);
            vis_data->stage_snapshots.clear();

            pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr last_curvatures(
                new pcl::PointCloud<pcl::PrincipalCurvatures>());

            int show_count = 0;
            for (const auto& step : pipeline_steps) {
                if (step.type == StepType::PassThrough) {
                    auto filter = std::make_shared<PassThroughFilter>(axis_idx_to_field(step.pass_axis_idx),
                                                                      step.pass_min, step.pass_max);
                    filter->apply(cloud_to_process);
                    continue;
                }

                if (step.type == StepType::VoxelGrid) {
                    auto filter = std::make_shared<VoxelGridFilter>(step.voxel_leaf_size);
                    filter->apply(cloud_to_process);
                    continue;
                }

                if (step.type == StepType::StatisticalOutlier) {
                    auto filter = std::make_shared<StatisticalOutlierFilter>(step.sor_k, step.sor_std_dev);
                    filter->apply(cloud_to_process);
                    continue;
                }

                if (step.type == StepType::ComputeCurvature) {
                    auto curvature_extractor = std::make_shared<CurvatureExtractor>(
                        step.curvature_ksearch, static_cast<double>(step.curvature_radius));
                    last_curvatures = curvature_extractor->extract(cloud_to_process);
                    continue;
                }

                if (is_display_step(step.type)) {
                    VisualizerData::StageSnapshot snapshot;
                    snapshot.cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud_to_process);
                    snapshot.normals.reset(new pcl::PointCloud<pcl::Normal>());
                    snapshot.display = step.display;

                    if (snapshot.display.show_normals) {
                        auto normal_extractor = std::make_shared<NormalExtractor>(
                            step.normal_ksearch, static_cast<double>(step.normal_radius));
                        snapshot.normals = normal_extractor->extract(cloud_to_process);
                    }

                    ++show_count;
                    snapshot.title = "View " + std::to_string(show_count) + ": " + step_type_to_label(step.type);
                    if (last_curvatures && !last_curvatures->empty()) {
                        snapshot.title += " (curvature ready)";
                    }

                    vis_data->stage_snapshots.push_back(snapshot);
                }
            }

            if (vis_data->stage_snapshots.empty()) {
                VisualizerData::StageSnapshot final_snapshot;
                final_snapshot.title = "View 1: Final Result";
                final_snapshot.cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud_to_process);
                final_snapshot.normals.reset(new pcl::PointCloud<pcl::Normal>());
                final_snapshot.display = make_default_step(StepType::ShowCloud).display;
                vis_data->stage_snapshots.push_back(final_snapshot);
            }

            vis_data->should_update = true;
            logger->log("CustomPipeline", vis_data->cloud_original->size(), cloud_to_process->size());
            std::cout << "[INFO] Pipeline executed. Final size: " << cloud_to_process->size() << "\n";
        }

        if (!cloud_loaded) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "请先在文件选择区加载点云文件。");
        }

        ImGui::End();

        // 显示关闭确认对话框
        if (g_request_exit) {
            ImGui::OpenPopup("关闭确认");
        }

        if (ImGui::BeginPopupModal("关闭确认", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("确定要关闭流水线控制面板吗？");
            ImGui::Spacing();

            float button_width = 120.0f;
            float available_width = ImGui::GetContentRegionAvail().x;
            float offset = (available_width - 2 * button_width - ImGui::GetStyle().ItemSpacing.x) / 2.0f;

            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);

            if (ImGui::Button("确定", ImVec2(button_width, 0))) {
                g_request_exit = false;
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                ImGui::CloseCurrentPopup();
            }

            ImGui::SetItemDefaultFocus();
            ImGui::SameLine();

            if (ImGui::Button("取消", ImVec2(button_width, 0))) {
                g_request_exit = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

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

    keep_visualizer_running = false;
    visualizer_thread.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
