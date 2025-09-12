#include "yolo_inference.h"

std::tuple<std::unique_ptr<YOLO_V8>, DL_INIT_PARAM> Initialize();
std::vector<DL_RESULT> Detector(std::unique_ptr<YOLO_V8>& p, const cv::Mat& img);
int ReadCocoYaml(std::unique_ptr<YOLO_V8>& p);