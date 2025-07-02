cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.14+

set(PROJECT_NAME Yolov8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# to show header files in Qt Creator
file(GLOB_RECURSE HEADER_FILES include/*.h)

# -------------- Google Test ------------------#
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- ONNXRuntime  ------------------#
set(ONNXRUNTIME_VERSION 1.21.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../hero_sam/onnxruntime-linux-x64-gpu-1.21.1")
include_directories(${ONNXRUNTIME_ROOT}/include)

# -------------- Cuda ------------------#
add_definitions(-DUSE_CUDA=1)
include_directories(/usr/local/cuda/include)

set(PROJECT_SOURCES
        src/main.cpp
        src/yolo_inference.cpp
        src/inference.cpp)

# Main executable (without tests)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Link libraries for main executable
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

# Test executable
add_executable(${PROJECT_NAME}_test
    test/yolo_test.cpp
    src/yolo_inference.cpp  # Include only the source files needed for testing (not main.cpp)
)

target_include_directories(${PROJECT_NAME}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Link libraries for test executable
target_link_libraries(${PROJECT_NAME}_test
    gtest_main
    gmock_main
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# Add test to CTest
add_test(NAME yolo_tests COMMAND ${PROJECT_NAME}_test)

# Download https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml
# and put it in the same folder of the executable file
configure_file(../hero_sam/yolo_inference/data/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)

# Copy yolov8n.onnx file to the same folder of the executable file
configure_file(../hero_sam/yolo_inference/model/yolo11m.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolo11m.onnx COPYONLY)

# Create folder name images in the same folder of the executable file
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)
