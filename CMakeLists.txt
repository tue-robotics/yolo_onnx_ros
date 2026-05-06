cmake_minimum_required(VERSION 3.5)

project(yolo_onnx_ros)

message(STATUS "Binary dir: ${CMAKE_CURRENT_BINARY_DIR}")

add_compile_options(-Wall -Wextra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------------------------
#                                        DEPENDENCIES
# ------------------------------------------------------------------------------------------------

find_package(OpenCV REQUIRED)
find_package(console_bridge REQUIRED)

# Optional CUDA support (disabled by default; enable with -DCUDA_ENABLED=ON)
option(CUDA_ENABLED "Enable CUDA/ONNX CUDA execution provider" OFF)
set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "")
set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "")
if(CUDA_ENABLED)
  find_package(CUDAToolkit REQUIRED)
  set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
  set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "CUDA::cudart")
endif()

# ONNX Runtime (set ONNXRUNTIME_ROOT if installed to a non-standard prefix)
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    "${ONNXRUNTIME_ROOT}/include"
    /usr/local/include/onnxruntime/core/session
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime/core/session
    /usr/include/onnxruntime
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS
    "${ONNXRUNTIME_ROOT}/lib"
    /usr/local/lib
    /usr/lib
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "Download from https://github.com/microsoft/onnxruntime/releases and extract, "
    "then re-run cmake with -DONNXRUNTIME_ROOT=<path>.")
endif()
message(STATUS "ONNX Runtime include dir : ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library     : ${ONNXRUNTIME_LIBRARY}")

# ------------------------------------------------------------------------------------------------
#                                     GENERATED HEADERS
# ------------------------------------------------------------------------------------------------

if(CUDA_ENABLED)
  set(yolo_onnx_ros_CUDA_ENABLED "TRUE")
else()
  set(yolo_onnx_ros_CUDA_ENABLED "FALSE")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/config.hpp.in
  ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp
  @ONLY
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  ${CMAKE_BINARY_DIR}/include
  SYSTEM
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${${PROJECT_NAME}_CUDA_INCLUDE_DIRS}
  ${console_bridge_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/detection.cpp
  src/yolo_inference.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${ONNXRUNTIME_LIBRARY}
  ${${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES}
  ${console_bridge_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(test_${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(test_${PROJECT_NAME}
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

# ------------------------------------------------------------------------------------------------
#                                        MODEL RESOURCES
# ------------------------------------------------------------------------------------------------

file(
  DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo11m/yolo11m.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo11m/yolo11m.onnx
  EXPECTED_HASH "SHA256=0eb68248f3868dfd8341e2e7d90e1e9d9594325e984574c11b0404f27673a419"
  SHOW_PROGRESS
)
file(
  DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo11m/coco.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo11m/coco.yaml
  EXPECTED_HASH "SHA256=721773c881cb7de92339c9e0f86dbf6e90b9ed551c1cf8d5bd6f043497e241aa"
  SHOW_PROGRESS
)

# ------------------------------------------------------------------------------------------------
#                                           INSTALL
# ------------------------------------------------------------------------------------------------

install(FILES
  ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp
  DESTINATION include/${PROJECT_NAME}
)
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  TARGETS test_${PROJECT_NAME}
  DESTINATION bin
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources
  DESTINATION share/${PROJECT_NAME}
)