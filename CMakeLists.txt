cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(yolo_onnx_ros)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Werror=all)
  add_compile_options(-Wextra -Werror=extra)
endif()

option(${PROJECT_NAME}_TENSORRT_ENABLED "Build TensorRT backend support" ON)

find_package(ament_cmake REQUIRED)
find_package(console_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(onnxruntime_ros REQUIRED)
# FindCUDAToolkit also finds a toolkit through /usr/local/cuda when nvcc isn't on the PATH,
# so check the compiler separately before enabling the TensorRT backend, which compiles CUDA code.
include(CheckLanguage)
check_language(CUDA)
find_package(CUDAToolkit QUIET)

if(${PROJECT_NAME}_TENSORRT_ENABLED AND CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
  set(${PROJECT_NAME}_TENSORRT_ENABLED TRUE)
  find_package(tensorrt_ros REQUIRED)
else()
  if(${PROJECT_NAME}_TENSORRT_ENABLED)
    message(STATUS "CUDA compiler or toolkit not found, building without the TensorRT backend")
  endif()
  set(${PROJECT_NAME}_TENSORRT_ENABLED FALSE)
endif()

if(onnxruntime_ros_CUDA_ENABLED OR ${PROJECT_NAME}_TENSORRT_ENABLED)
  set(${PROJECT_NAME}_CUDA_ENABLED TRUE)
else()
  set(${PROJECT_NAME}_CUDA_ENABLED FALSE)
endif()

if(${PROJECT_NAME}_CUDA_ENABLED)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "onnxruntime_ros is built with CUDA, but the CUDA toolkit could not be found")
  endif()
  set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
endif()

if(${PROJECT_NAME}_TENSORRT_ENABLED AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64" AND
    (EXISTS "/etc/nv_tegra_release" OR EXISTS "/usr/lib/aarch64-linux-gnu/tegra"))
    set(CMAKE_CUDA_ARCHITECTURES "72;87")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
  endif()
endif()

# Only the TensorRT backend compiles CUDA code; the ONNX Runtime CUDA path just needs the toolkit.
if(${PROJECT_NAME}_TENSORRT_ENABLED)
  enable_language(CUDA)
endif()

set(${PROJECT_NAME}_YOLOS_TRT_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/YOLOs-CPP-TensorRT")
set(${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR
  "${${PROJECT_NAME}_YOLOS_TRT_ROOT}/include")
set(${PROJECT_NAME}_YOLOS_TRT_HEADER
    "${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}/yolos/tasks/detection.hpp")
set(${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE
    "${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}/yolos/core/cuda_preprocessing.cu")
if(${PROJECT_NAME}_TENSORRT_ENABLED AND
  (NOT EXISTS "${${PROJECT_NAME}_YOLOS_TRT_HEADER}" OR
  NOT EXISTS "${${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE}"))
  message(FATAL_ERROR
    "${PROJECT_NAME}_TENSORRT_ENABLED=ON requires the YOLOs-CPP-TensorRT submodule. "
    "Expected ${${PROJECT_NAME}_YOLOS_TRT_HEADER} and "
    "${${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE}. Initialize the submodule or disable TensorRT support.")
endif()

set(${PROJECT_NAME}_CONFIG_HEADER ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME})
set(YOLO_ONNX_ROS_CUDA_ENABLED ${${PROJECT_NAME}_CUDA_ENABLED})
set(YOLO_ONNX_ROS_TENSORRT_ENABLED ${${PROJECT_NAME}_TENSORRT_ENABLED})
configure_file(include/${PROJECT_NAME}/config.hpp.in ${${PROJECT_NAME}_CONFIG_HEADER} @ONLY)

add_library(${PROJECT_NAME} SHARED
  src/detection.cpp
  src/yolo_inference.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${${PROJECT_NAME}_CUDA_INCLUDE_DIRS}
  ${console_bridge_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
  console_bridge::console_bridge
)
target_link_libraries(${PROJECT_NAME} PUBLIC
  onnxruntime_ros::onnxruntime_ros
  ${OpenCV_LIBRARIES}
)
if(${PROJECT_NAME}_CUDA_ENABLED)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
endif()

if(${PROJECT_NAME}_TENSORRT_ENABLED)
  find_package(Threads REQUIRED)
  # Vendored source, keep it out of compile_commands.json so clang-tidy does not check it
  set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
  add_library(${PROJECT_NAME}_cuda_preprocessing STATIC ${${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE})
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set_target_properties(${PROJECT_NAME}_cuda_preprocessing PROPERTIES
    COMPILE_OPTIONS ""
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(${PROJECT_NAME}_cuda_preprocessing SYSTEM PRIVATE
    ${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}
  )
  target_link_libraries(${PROJECT_NAME}_cuda_preprocessing PRIVATE CUDA::cudart)
  # detection.hpp (public) includes the yolos headers, which include NvInfer.h
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
    $<BUILD_INTERFACE:${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}>
  )
  target_link_libraries(${PROJECT_NAME} PUBLIC
    tensorrt_ros::tensorrt_ros
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}_cuda_preprocessing
    Threads::Threads
  )
  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node PRIVATE ${PROJECT_NAME} ${OpenCV_LIBRARIES})

set(${PROJECT_NAME}_DOWNLOAD_ARGS SHOW_PROGRESS)
if(NOT "$ENV{GITHUB_TOKEN}" STREQUAL "")
  list(APPEND ${PROJECT_NAME}_DOWNLOAD_ARGS HTTPHEADER "Authorization: ******")
endif()

if(${PROJECT_NAME}_TENSORRT_ENABLED)
  file(DOWNLOAD
    https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo26m/yolo26m.trt
    ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo26m/yolo26m.trt
    EXPECTED_HASH "SHA256=8b605d40955ede9bc49221fe4096a8e64e66dc1503b50322f229ed0e6dee0c43"
    ${${PROJECT_NAME}_DOWNLOAD_ARGS}
  )
  file(DOWNLOAD
    https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo26m/coco.names
    ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo26m/coco.names
    EXPECTED_HASH "SHA256=bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a"
    ${${PROJECT_NAME}_DOWNLOAD_ARGS}
  )
endif()

if(NOT ${PROJECT_NAME}_TENSORRT_ENABLED OR BUILD_TESTING)
  file(DOWNLOAD
    https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo26m/yolo26m.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo26m/yolo26m.onnx
    EXPECTED_HASH "SHA256=17d9d4491c8954d69ce95bc8f707d883cae4a1b6508ee9f4dca8e0b38b1addc9"
    ${${PROJECT_NAME}_DOWNLOAD_ARGS}
  )
  file(DOWNLOAD
    https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo26m/coco.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo26m/coco.yaml
    EXPECTED_HASH "SHA256=721773c881cb7de92339c9e0f86dbf6e90b9ed551c1cf8d5bd6f043497e241aa"
    ${${PROJECT_NAME}_DOWNLOAD_ARGS}
  )
endif()

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  PATTERN "config.hpp.in" EXCLUDE
)
install(FILES ${${PROJECT_NAME}_CONFIG_HEADER} DESTINATION include/${PROJECT_NAME})

if(${PROJECT_NAME}_TENSORRT_ENABLED)
  install(DIRECTORY ${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}/yolos DESTINATION include)
  install(FILES ${${PROJECT_NAME}_YOLOS_TRT_ROOT}/LICENSE
    DESTINATION share/${PROJECT_NAME}/3rdparty/YOLOs-CPP-TensorRT)
endif()

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_clang_tidy REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)
  find_package(tue_lint_config REQUIRED)
  ament_clang_format(CONFIG_FILE ${tue_lint_config_DIR}/../config/.clang-format --clang-format-version=21)
  ament_clang_tidy(CONFIG_FILE ${tue_lint_config_DIR}/../config/.clang-tidy --clang-tidy-version=21
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/compile_commands.json)
  ament_lint_cmake(MAX_LINE_LENGTH 120 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
  ament_xmllint(MAX_LINE_LENGTH 120)

  ament_add_gtest(yolo_test test/yolo_test.cpp
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/resources/yolo26m")
  if(TARGET yolo_test)
    target_link_libraries(yolo_test
      ${PROJECT_NAME}
      ${OpenCV_LIBRARIES}
    )
  endif()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV onnxruntime_ros)
if(${PROJECT_NAME}_CUDA_ENABLED)
  ament_export_dependencies(CUDAToolkit)
endif()
if(${PROJECT_NAME}_TENSORRT_ENABLED)
  ament_export_dependencies(tensorrt_ros)
endif()

ament_package()
