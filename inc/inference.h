#include "yolo_inference.h"

std::unique_ptr<YOLO_V8> Initialize();
std::vector<DL_RESULT> DetectObjects(std::unique_ptr<YOLO_V8>& p, const cv::Mat& img);
int ReadCocoYaml(std::unique_ptr<YOLO_V8>& p);