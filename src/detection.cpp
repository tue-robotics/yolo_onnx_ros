#include "yolo_onnx_ros/detection.hpp"
#include <yolo_onnx_ros/config.hpp>

#include <console_bridge/console.h>

#include <fstream>
#include <iomanip>
#include <iostream>

std::vector<DL_RESULT> Detector(YoloWrapper& wrapper, const cv::Mat& img)
{
    std::vector<DL_RESULT> res;

    if (wrapper.backend == YOLO::Backend::kOnnx)
    {
        wrapper.onnxDetector->RunSession(img, res);
#ifdef LOGGING
        for (auto& re : res)
        {
            cv::RNG rng(cv::getTickCount());
            cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

            cv::rectangle(img, re.box, color, 3);

            float confidence = floor(100 * re.confidence) / 100;
            std::cout << std::fixed << std::setprecision(2);

            std::string label;
            if (!wrapper.classes.empty() && re.classId >= 0 &&
                static_cast<size_t>(re.classId) < wrapper.classes.size())
            {
                label = wrapper.classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
            }

            cv::rectangle(
                img,
                cv::Point(re.box.x, re.box.y - 25),
                cv::Point(re.box.x + static_cast<int>(label.length()) * 15, re.box.y),
                color,
                cv::FILLED
            );

            cv::putText(
                img,
                label,
                cv::Point(re.box.x, re.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 0, 0),
                2
            );
        }
#endif
    }
#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
    else if (wrapper.backend == YOLO::Backend::kTensorRT)
    {
        auto detections = wrapper.trtDetector->detect(img, 0.1f, 0.5f);
        res.reserve(detections.size());
        for (const auto& det : detections)
        {
            DL_RESULT r;
            r.classId    = det.classId;
            r.confidence = det.conf;
            r.box        = cv::Rect(det.box.x, det.box.y, det.box.width, det.box.height);
            res.push_back(r);
        }
    }
#else
    else if (wrapper.backend == YOLO::Backend::kTensorRT)
    {
        throw std::runtime_error(
            "[ERROR] Detector: backend 'tensorRT' was requested but "
            "'yolo_onnx_ros' was compiled without TensorRT support. "
            "Rebuild against a CUDA-enabled onnxruntime_ros package.");
    }
#endif

    return res;
}



// void Classifier(std::unique_ptr<YOLO_V8>& p)
// {
//     std::filesystem::path current_path = std::filesystem::current_path();
//     std::filesystem::path imgs_path = current_path;// / "images"
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int> dis(0, 255);
//     for (auto& i : std::filesystem::directory_iterator(imgs_path))
//     {
//         if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
//         {
//             std::string img_path = i.path().string();
//             //std::cout << img_path << std::endl;
//             cv::Mat img = cv::imread(img_path);
//             std::vector<DL_RESULT> res;
//             const char* ret = p->RunSession(img, res);

//             float positionY = 50;
//             for (int i = 0; i < res.size(); i++)
//             {
//                 int r = dis(gen);
//                 int g = dis(gen);
//                 int b = dis(gen);
//                 cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
//                 cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
//                 positionY += 50;
//             }

//             cv::imshow("TEST_CLS", img);
//             cv::waitKey(0);
//             cv::destroyAllWindows();
//         }

//     }
// }



int ReadYaml(const std::filesystem::path& filename, std::unique_ptr<YOLO_V8>& p)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        CONSOLE_BRIDGE_logError("[ReadYaml] Failed to open: %s", filename.c_str());
        return 1;
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = lines.size();
    bool in_names_section = false;

    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
            in_names_section = true;
        }
        else if (in_names_section && !lines[i].empty() &&
                 lines[i][0] != ' ' && lines[i][0] != '\t' && lines[i][0] != '-')
        {
            // Found next top-level key (no indentation)
            end = i;
            break;
        }
    }

    if (!in_names_section || start >= lines.size())
    {
        CONSOLE_BRIDGE_logError("[ReadYaml] Could not find 'names:' section in %s", filename.c_str());
        return 1;
    }

    // Extract the names with proper trimming
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::string trimmed = lines[i];
        // Remove leading whitespace
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));

        if (trimmed.empty()) continue;

        // Handle both formats: "0: Apple" and "- Apple"
        std::string name;
        size_t colon_pos = trimmed.find(':');
        if (colon_pos != std::string::npos) {
            // Dict format: "0: Apple"
            name = trimmed.substr(colon_pos + 1);
        } else if (trimmed[0] == '-') {
            // List format: "- Apple"
            name = trimmed.substr(1);
        } else {
            continue;
        }

        // Trim whitespace from extracted name
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t\r\n") + 1);

        if (!name.empty()) {
            names.push_back(name);
        }
    }

    if (names.empty())
    {
        CONSOLE_BRIDGE_logError("[ReadYaml] No class names found in %s", filename.c_str());
        return 1;
    }

    p->classes = names;
    CONSOLE_BRIDGE_logInform("[ReadYaml] Loaded %u classes from %s", names.size(), filename.c_str());
    return 0;
}

std::tuple<YoloWrapper, DL_INIT_PARAM> Initialize(const std::filesystem::path& model_filename,
                                                   YOLO::Backend backend)
{
    YoloWrapper wrapper;
    wrapper.backend = backend;
    DL_INIT_PARAM params;

    if (backend == YOLO::Backend::kOnnx)
    {
        wrapper.onnxDetector = std::make_unique<YOLO_V8>();

        // Class names are optional — skip silently if absent (e.g. YOLO26)
        const auto yaml_path = model_filename.parent_path() / "coco.yaml";
        if (std::filesystem::exists(yaml_path))
            ReadYaml(yaml_path, wrapper.onnxDetector);

        params.rectConfidenceThreshold = 0.1;
        params.iouThreshold = 0.5;
        params.modelPath = model_filename;
        params.imgSize = { 640, 640 };
#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
        params.cudaEnable = true;
        params.modelType  = YOLO_DETECT_V8;
#else
        params.cudaEnable = false;
        params.modelType  = YOLO_DETECT_V8;
#endif
        wrapper.onnxDetector->CreateSession(params);
        wrapper.classes = wrapper.onnxDetector->classes;
    }
#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
    else if (backend == YOLO::Backend::kTensorRT)
    {
        // Class names are optional — pass empty string when absent (e.g. YOLO26)
        const auto names_path = model_filename.parent_path() / "coco.names";
        const std::string labels = std::filesystem::exists(names_path) ? names_path.string() : "";
        wrapper.trtDetector = std::make_unique<yolos::det::YOLODetector>(model_filename.string(), labels);
        wrapper.classes = wrapper.trtDetector->getClassNames();
    }
#else
    else if (backend == YOLO::Backend::kTensorRT)
    {
        throw std::runtime_error(
            "[ERROR] Initialize: backend 'tensorRT' was requested but "
            "'yolo_onnx_ros' was compiled WITHOUT TensorRT support. "
            "Rebuild with -DCUDA_ENABLED=ON.");
    }
#endif

    return {std::move(wrapper), std::move(params)};
}


// void ClsTest()
// {
//     std::unique_ptr<YOLO_V8> yoloDetector = std::make_unique<YOLO_V8>();
//     std::string model_path = "cls.onnx";
//     ReadCocoYaml(yoloDetector);
//     DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
//     yoloDetector->CreateSession(params);
//     Classifier(yoloDetector);
// }
