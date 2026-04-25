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
    bool save_output = false;  // 是否保存该步骤的输出

    // PassThrough
    int pass_axis_idx = 2;  // 0:x, 1:y, 2:z
    float pass_min = -100.0F;
    float pass_max = 100.0F;

    // VoxelGrid
    float voxel_leaf_size = 0.01F;

    // StatisticalOutlier
    int sor_k = 50;
    float sor_std_dev = 1.0F;

    // Curvature
    int curvature_ksearch = 10;
    float curvature_radius = 0.0F;

    // Normals for display
    int normal_ksearch = 10;
    float normal_display_length = 0.05F;  // 法线显示长度（仅用于可视化，不影响计算）

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

// ================================================================
// 文件保存辅助函数
// ================================================================
std::string get_output_folder(const std::string& original_file_path) {
    if (original_file_path.empty()) return "";

    // 在原文件所在目录下创建 output 子文件夹
    std::string dir = fs::path(original_file_path).parent_path().string();
    if (dir.empty()) dir = ".";

    std::string output_dir = dir + "/output";
    fs::create_directories(output_dir);
    return output_dir;
}

std::string generate_output_filename(const std::string& output_folder, int step_number,
                                     const std::string& original_filename) {
    if (output_folder.empty()) return "";

    // 获取原文件的扩展名
    std::string ext = fs::path(original_filename).extension().string();
    if (ext == ".bin") {
        ext = ".pcd";
    } else if (ext != ".pcd" && ext != ".ply") {
        ext = ".pcd";
    }

    // 生成文件名：output_folder/filename_XN.ext
    // 例如：output/cloud_01.pcd, output/cloud_02.ply
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/cloud_%02d%s", output_folder.c_str(), step_number, ext.c_str());
    return std::string(filename);
}

// 将 XYZ 点云与法线/曲率特征合并为 PointNormal 格式。
// prev_featured_cloud: 先前保存过的特征云，用于累积之前计算过的特征数据。
// 注意：prev_featured_cloud 只在点索引一致的场景下有效（即流水线中未插入滤波步骤）。
pcl::PointCloud<pcl::PointNormal>::Ptr build_featured_point_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, const pcl::PointCloud<pcl::Normal>::ConstPtr& normals,
    const pcl::PointCloud<pcl::PrincipalCurvatures>::ConstPtr& curvatures,
    const pcl::PointCloud<pcl::PointNormal>::ConstPtr& prev_featured_cloud = nullptr) {
    pcl::PointCloud<pcl::PointNormal>::Ptr featured_cloud(new pcl::PointCloud<pcl::PointNormal>());
    if (!cloud || cloud->empty()) {
        return featured_cloud;
    }

    featured_cloud->reserve(cloud->size());
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        const auto& source_point = cloud->points[i];
        pcl::PointNormal target_point;
        // 如果有先前特征云且索引匹配，先拷贝它（包含历史法线/曲率），再覆盖最新的数据
        if (prev_featured_cloud && i < prev_featured_cloud->size()) {
            target_point = prev_featured_cloud->points[i];
        }
        target_point.x = source_point.x;
        target_point.y = source_point.y;
        target_point.z = source_point.z;

        if (normals && i < normals->size()) {
            target_point.normal_x = normals->points[i].normal_x;
            target_point.normal_y = normals->points[i].normal_y;
            target_point.normal_z = normals->points[i].normal_z;
        }

        if (curvatures && i < curvatures->size()) {
            target_point.curvature = curvatures->points[i].pc1;
        }

        featured_cloud->push_back(target_point);
    }

    return featured_cloud;
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
            io.FontGlobalScale = 1.0F;
            return true;
        }
    }

    io.Fonts->AddFontDefault();
    io.FontGlobalScale = 1.30F;
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

void render_display_config_ui(DisplayConfig& config) {
    ImGui::SliderInt("点大小", &config.point_size, 1, 8);
    ImGui::SliderInt("点云颜色 R", &config.color_r, 0, 255);
    ImGui::SliderInt("点云颜色 G", &config.color_g, 0, 255);
    ImGui::SliderInt("点云颜色 B", &config.color_b, 0, 255);

    ImGui::Checkbox("显示坐标轴", &config.show_coordinate_axis);
    if (config.show_coordinate_axis) {
        ImGui::SliderFloat("坐标轴缩放", &config.axis_scale, 0.02F, 1.0F, "%.2F");
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
int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1650, 1500, "Point Cloud Pipeline Control", NULL, NULL);
    if (window == NULL) return 1;

    glfwSetWindowCloseCallback(window, glfw_window_close_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui::GetStyle().ScaleAllSizes(1.15F);
    std::string font_loaded_path;
    const bool has_chinese_font = setup_imgui_font(24.0F, font_loaded_path);

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

    char search_dir[512] = "./";

    std::vector<PipelineStepConfig> pipeline_steps = {
        make_default_step(StepType::PassThrough),      make_default_step(StepType::VoxelGrid),
        make_default_step(StepType::ComputeCurvature), make_default_step(StepType::ShowCloud),
        make_default_step(StepType::ShowNormals),
    };

    int add_step_type_idx = static_cast<int>(StepType::ShowCloud);

    auto logger = std::make_shared<ProcessingLog>("../logs/processing_gui.log");

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
            ImGui::TextColored(ImVec4(1, 0.2F, 0.2F, 1), "路径不可用或不是文件夹。请检查搜索文件夹路径。");
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

        // 使用Columns实现水平布局：最多5列（每行显示5个步骤）
        int max_cols = 5;
        int num_cols = std::min(max_cols, static_cast<int>(pipeline_steps.size()));
        if (num_cols > 0) {
            ImGui::Columns(num_cols, "PipelineStepsColumns", false);
            ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / num_cols - 10);
            for (int c = 1; c < num_cols; ++c) {
                ImGui::SetColumnWidth(c, ImGui::GetWindowWidth() / num_cols - 10);
            }

            for (std::size_t i = 0; i < pipeline_steps.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));

                // 每个步骤的边框
                ImGui::BeginChild(("Step##" + std::to_string(i)).c_str(), ImVec2(-1, 400), true);

                ImGui::Text("步骤 %d", static_cast<int>(i + 1));

                int current_type = static_cast<int>(pipeline_steps[i].type);
                ImGui::SetNextItemWidth(200);
                if (ImGui::Combo(("类型##" + std::to_string(i)).c_str(), &current_type, step_labels,
                                 IM_ARRAYSIZE(step_labels))) {
                    pipeline_steps[i] = make_default_step(static_cast<StepType>(current_type));
                }

                ImGui::Spacing();
                float button_width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3;

                if (ImGui::Button(("上##" + std::to_string(i)).c_str(), ImVec2(button_width, 0)) && i > 0) {
                    std::swap(pipeline_steps[i], pipeline_steps[i - 1]);
                }
                ImGui::SameLine();
                if (ImGui::Button(("下##" + std::to_string(i)).c_str(), ImVec2(button_width, 0)) &&
                    i + 1 < pipeline_steps.size()) {
                    std::swap(pipeline_steps[i], pipeline_steps[i + 1]);
                }
                ImGui::SameLine();
                if (ImGui::Button(("删除##" + std::to_string(i)).c_str(), ImVec2(button_width, 0)) &&
                    pipeline_steps.size() > 2) {
                    pipeline_steps.erase(pipeline_steps.begin() + static_cast<std::ptrdiff_t>(i));
                    ImGui::EndChild();
                    ImGui::PopID();
                    ImGui::NextColumn();
                    continue;
                }

                ImGui::Spacing();
                ImGui::Separator();

                const StepType type = pipeline_steps[i].type;

                if (type == StepType::PassThrough) {
                    const char* axis_labels[] = {"x", "y", "z"};
                    ImGui::SetNextItemWidth(120);
                    ImGui::Combo(("轴##" + std::to_string(i)).c_str(), &pipeline_steps[i].pass_axis_idx, axis_labels,
                                 IM_ARRAYSIZE(axis_labels));
                    ImGui::SetNextItemWidth(200);
                    ImGui::DragFloatRange2(("最小/最大##" + std::to_string(i)).c_str(), &pipeline_steps[i].pass_min,
                                           &pipeline_steps[i].pass_max, 0.2f, -200.0f, 200.0f);
                } else if (type == StepType::VoxelGrid) {
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderFloat(("体素大小##" + std::to_string(i)).c_str(), &pipeline_steps[i].voxel_leaf_size,
                                       0.001F, 0.5F, "%.3f m");
                } else if (type == StepType::StatisticalOutlier) {
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderInt(("均值K值##" + std::to_string(i)).c_str(), &pipeline_steps[i].sor_k, 1, 200);
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderFloat(("标准差##" + std::to_string(i)).c_str(), &pipeline_steps[i].sor_std_dev, 0.1F,
                                       5.0F, "%.2f");
                } else if (type == StepType::ComputeCurvature) {
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderInt(("曲率K搜索##" + std::to_string(i)).c_str(), &pipeline_steps[i].curvature_ksearch,
                                     3, 100);
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderFloat(("曲率半径##" + std::to_string(i)).c_str(), &pipeline_steps[i].curvature_radius,
                                       0.0F, 1.0F, "%.3f");
                    ImGui::TextWrapped("半径 > 0 时使用半径搜索，否则使用 KSearch。");
                } else if (type == StepType::ShowCloud) {
                    ImGui::TextWrapped("点云显示模块");
                    ImGui::TextWrapped("颜色/点大小等在下方显示参数区。");
                } else if (type == StepType::ShowNormals) {
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderInt(("法线K搜索##" + std::to_string(i)).c_str(), &pipeline_steps[i].normal_ksearch, 3,
                                     100);
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderFloat(("法线长度##" + std::to_string(i)).c_str(),
                                       &pipeline_steps[i].normal_display_length, 0.01f, 0.5f, "%.3f");
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderInt(("法线密度##" + std::to_string(i)).c_str(),
                                     &pipeline_steps[i].display.normal_level, 1, 100);
                }

                ImGui::Spacing();
                ImGui::Separator();

                // 保存输出选项（仅显示非ShowCloud步骤）
                if (type != StepType::ShowCloud) {
                    ImGui::Checkbox(("保存输出##" + std::to_string(i)).c_str(), &pipeline_steps[i].save_output);
                    ImGui::TextWrapped("勾选则在output文件夹保存此步骤的输出。");
                }

                ImGui::EndChild();
                ImGui::PopID();
                ImGui::NextColumn();
            }

            ImGui::Columns(1);
        }

        ImGui::Separator();
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("新增步骤类型", &add_step_type_idx, step_labels, IM_ARRAYSIZE(step_labels));
        if (ImGui::Button("+ 添加步骤", ImVec2(-1, 30))) {
            pipeline_steps.push_back(make_default_step(static_cast<StepType>(add_step_type_idx)));
        }

        // 3) 可视化部分
        ImGui::SeparatorText("原始点云的可视化部分");
        ImGui::TextWrapped(
            "原始点云固定有一个视口。流水线里每个显示步骤会新增一个视口，并按该步骤中的显示参数渲染"
            "（颜色、点大小、是否显示法线、法线密度/长度、是否显示坐标轴、背景色）。");

        if (ImGui::CollapsingHeader("原始点云显示参数", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_display_config_ui(vis_data->original_display);
        }

        int display_step_count = 0;
        for (const auto& step : pipeline_steps) {
            if (is_display_step(step.type)) {
                ++display_step_count;
            }
        }
        ImGui::Text("预计视口数量：%d（1 个原始 + %d 个显示步骤）", 1 + display_step_count, display_step_count);

        if (ImGui::Button("执行流水线", ImVec2(-1, 48)) && cloud_loaded) {
            std::lock_guard<std::mutex> lock(vis_data->mutex);

            auto cloud_to_process = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*vis_data->cloud_original);
            vis_data->stage_snapshots.clear();

            // 获取输出文件夹路径
            std::string output_folder = get_output_folder(current_file_path);

            // ============================================
            // 使用Pipeline类来构建和执行流水线处理
            // ============================================
            auto pipeline = std::make_shared<PointCloudPipeline>(logger);

            int show_count = 0;
            int step_counter = 0;
            bool has_pending_stages = false;
            // 标记当前 pipeline 中是否有滤波步骤。滤波会移除/改变点云索引，
            // 使得之前保存的 feature cloud（含法线/曲率数据）与新点云的点索引不匹配，
            // 因此在 flush 执行后必须废弃 previous_feature_cloud。
            bool filters_pending_in_pipeline = false;
            pcl::PointCloud<pcl::PointNormal>::Ptr previous_feature_cloud(new pcl::PointCloud<pcl::PointNormal>());
            bool previous_feature_cloud_valid = false;

            // 刷新：将累积的 pipeline 步骤（滤波+特征提取）执行到 cloud_to_process 上
            auto flush_pending_pipeline = [&]() {
                if (!has_pending_stages) {
                    return;
                }
                pipeline->execute(cloud_to_process);
                has_pending_stages = false;
                // 如果有滤波步骤执行过，点索引已改变，废弃旧的 feature cloud
                if (filters_pending_in_pipeline) {
                    previous_feature_cloud_valid = false;
                    filters_pending_in_pipeline = false;
                }
            };

            auto write_step_output =
                [&](int step_number, const pcl::PointCloud<pcl::Normal>::ConstPtr& normals = nullptr,
                    const pcl::PointCloud<pcl::PrincipalCurvatures>::ConstPtr& curvatures = nullptr) {
                    std::string output_file = generate_output_filename(output_folder, step_number, current_file_name);
                    if (output_file.empty()) {
                        return;
                    }
                    const pcl::PointCloud<pcl::PointNormal>::ConstPtr prev_cloud =
                        previous_feature_cloud_valid ? previous_feature_cloud : nullptr;
                    auto featured_cloud = build_featured_point_cloud(cloud_to_process, normals, curvatures, prev_cloud);
                    const bool use_feature_cloud = previous_feature_cloud_valid || normals || curvatures;

                    bool saved = false;
                    if (use_feature_cloud) {
                        saved = PointCloudIO::save(output_file, *featured_cloud);
                        if (saved) {
                            previous_feature_cloud = featured_cloud;
                            previous_feature_cloud_valid = true;
                        }
                    } else {
                        saved = PointCloudIO::save(output_file, *cloud_to_process);
                    }

                    if (saved) {
                        std::cout << "[INFO] Saved step " << step_number << " to " << output_file << "\n";
                        if (logger) {
                            logger->log("Save step " + std::to_string(step_number), cloud_to_process->size(),
                                        cloud_to_process->size());
                        }
                    } else {
                        std::cout << "[WARN] Failed to save step " << step_number << "\n";
                    }
                };

            auto capture_display_snapshot = [&](const PipelineStepConfig& step, int step_number) {
                // 显示步骤作为断点：先把前面的滤波/特征提取执行完，再截取当前结果
                flush_pending_pipeline();

                VisualizerData::StageSnapshot snapshot;
                snapshot.cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud_to_process);
                snapshot.normals.reset(new pcl::PointCloud<pcl::Normal>());
                snapshot.display = step.display;

                const auto normals = pipeline->getNormals();
                const auto curvatures = pipeline->getCurvatures();

                if (snapshot.display.show_normals && normals && !normals->empty()) {
                    snapshot.normals = normals;
                    snapshot.display.normal_scale = step.normal_display_length;
                }

                if (step.save_output) {
                    const pcl::PointCloud<pcl::Normal>::ConstPtr normals_for_save =
                        (snapshot.display.show_normals && normals && !normals->empty()) ? normals : nullptr;
                    write_step_output(step_number, normals_for_save, curvatures);
                }

                ++show_count;
                snapshot.title = "View " + std::to_string(show_count) + ": " + step_type_to_label(step.type);
                if (curvatures && !curvatures->empty()) {
                    snapshot.title += " (curvature ready)";
                }

                vis_data->stage_snapshots.push_back(snapshot);
                pipeline = std::make_shared<PointCloudPipeline>(logger);
            };

            for (const auto& step : pipeline_steps) {
                ++step_counter;

                // 为过滤和特征提取步骤添加到Pipeline
                // 滤波步骤：添加到 pipeline，标记 filters_pending_in_pipeline
                // 以便后续 flush 时废弃旧的 feature cloud
                if (step.type == StepType::PassThrough) {
                    auto filter = std::make_shared<PassThroughFilter>(axis_idx_to_field(step.pass_axis_idx),
                                                                      step.pass_min, step.pass_max);
                    pipeline->addStage(filter);
                    has_pending_stages = true;
                    filters_pending_in_pipeline = true;
                    if (step.save_output) {
                        flush_pending_pipeline();  // 此 flush 会因 filters_pending 而废弃 previous_feature_cloud
                        write_step_output(step_counter, nullptr, nullptr);
                        pipeline = std::make_shared<PointCloudPipeline>(logger);
                    }
                } else if (step.type == StepType::VoxelGrid) {
                    auto filter = std::make_shared<VoxelGridFilter>(step.voxel_leaf_size);
                    pipeline->addStage(filter);
                    has_pending_stages = true;
                    filters_pending_in_pipeline = true;
                    if (step.save_output) {
                        flush_pending_pipeline();
                        write_step_output(step_counter, nullptr, nullptr);
                        pipeline = std::make_shared<PointCloudPipeline>(logger);
                    }
                } else if (step.type == StepType::StatisticalOutlier) {
                    auto filter = std::make_shared<StatisticalOutlierFilter>(step.sor_k, step.sor_std_dev);
                    pipeline->addStage(filter);
                    has_pending_stages = true;
                    filters_pending_in_pipeline = true;
                    if (step.save_output) {
                        flush_pending_pipeline();
                        write_step_output(step_counter, nullptr, nullptr);
                        pipeline = std::make_shared<PointCloudPipeline>(logger);
                    }
                } else if (step.type == StepType::ComputeCurvature) {
                    auto curvature_extractor = std::make_shared<CurvatureExtractor>(
                        step.curvature_ksearch, static_cast<double>(step.curvature_radius));
                    pipeline->setCurvatureExtractor(curvature_extractor);
                    has_pending_stages = true;
                    if (step.save_output) {
                        flush_pending_pipeline();
                        write_step_output(step_counter, nullptr, pipeline->getCurvatures());
                        pipeline = std::make_shared<PointCloudPipeline>(logger);
                    }
                } else if (step.type == StepType::ShowNormals) {
                    auto normal_extractor = std::make_shared<NormalExtractor>(step.normal_ksearch, 0.0);
                    pipeline->setNormalExtractor(normal_extractor);
                    has_pending_stages = true;
                    capture_display_snapshot(step, step_counter);
                } else if (is_display_step(step.type)) {
                    capture_display_snapshot(step, step_counter);
                }
            }

            // 执行最后的Pipeline步骤
            if (has_pending_stages) {
                pipeline->execute(cloud_to_process);
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

        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5F, ImGui::GetIO().DisplaySize.y * 0.5F),
                                ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));

        if (ImGui::BeginPopupModal("关闭确认", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("确定要关闭流水线控制面板吗？");
            ImGui::Spacing();

            float button_width = 120.0F;
            float available_width = ImGui::GetContentRegionAvail().x;
            float offset = (available_width - 2 * button_width - ImGui::GetStyle().ItemSpacing.x) / 2.0F;

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
