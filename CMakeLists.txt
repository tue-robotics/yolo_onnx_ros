cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME yolo_onnx_ros)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp

)
find_package(OpenCV REQUIRED)

# ONNX Runtime (adjust to your local path)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../hero_sam.bak/onnxruntime-linux-x64-gpu-1.21.1")
include_directories(${ONNXRUNTIME_ROOT}/include)

# CUDA (optional)
add_definitions(-DUSE_CUDA=1)
include_directories(/usr/local/cuda/include)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_core
  CATKIN_DEPENDS roscpp
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Core library (to be linked by other packages)
add_library(${PROJECT_NAME}_core
  src/yolo_inference.cpp
  src/detection.cpp
)
target_link_libraries(${PROJECT_NAME}_core
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# Node executable
add_executable(${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_core
)

# Install headers, library, and node so other packages can depend/link
install(TARGETS ${PROJECT_NAME}_core ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# Runtime assets (# Download https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)
configure_file(/home/amigo/Documents/repos/hero_sam.bak/yolo_inference/data/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)
configure_file(/home/amigo/Documents/repos/hero_sam.bak/yolo_inference/model/yolo11m.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolo11m.onnx COPYONLY)