#include "yolo_inference.h"

std::vector<DL_RESULT> Detect(const cv::Mat& img);
std::vector<DL_RESULT> DetectObjects(std::unique_ptr<YOLO_V8>& p, const cv::Mat& img);
int ReadCocoYaml(std::unique_ptr<YOLO_V8>& p);