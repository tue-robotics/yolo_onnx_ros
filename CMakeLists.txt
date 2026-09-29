cmake_minimum_required(VERSION 3.5)

project(yolo_onnx_ros)

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror=all>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror=extra>
)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(YOLO_ONNX_ROS_ENABLE_TENSORRT "Build TensorRT backend support" OFF)

find_package(catkin REQUIRED
  COMPONENTS
  onnxruntime_ros
)

set(${PROJECT_NAME}_CATKIN_DEPENDS onnxruntime_ros)

if(YOLO_ONNX_ROS_ENABLE_TENSORRT)
  find_package(tensorrt_ros REQUIRED)

  set(${PROJECT_NAME}_YOLOS_TRT_ROOT
      "${CMAKE_CURRENT_SOURCE_DIR}/include/3rdparty/YOLOs-CPP-TensorRT")
  set(${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR
      "${${PROJECT_NAME}_YOLOS_TRT_ROOT}/include")
  set(${PROJECT_NAME}_YOLOS_TRT_HEADER
      "${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}/yolos/tasks/detection.hpp")
  set(${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE
      "${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}/yolos/core/cuda_preprocessing.cu")
  if(NOT EXISTS "${${PROJECT_NAME}_YOLOS_TRT_HEADER}" OR
     NOT EXISTS "${${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE}")
    message(FATAL_ERROR
      "YOLO_ONNX_ROS_ENABLE_TENSORRT=ON requires the YOLOs-CPP-TensorRT submodule. "
      "Expected ${${PROJECT_NAME}_YOLOS_TRT_HEADER} and "
      "${${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE}. Initialize the submodule or disable TensorRT support.")
  endif()

  list(APPEND ${PROJECT_NAME}_CATKIN_DEPENDS tensorrt_ros)
  list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR})
endif()

find_package(console_bridge REQUIRED)
find_package(OpenCV REQUIRED)

set(${PROJECT_NAME}_CUDA_ENABLED ${onnxruntime_ros_CUDA_ENABLED})
if(YOLO_ONNX_ROS_ENABLE_TENSORRT)
  set(${PROJECT_NAME}_TENSORRT_ENABLED TRUE)
else()
  set(${PROJECT_NAME}_TENSORRT_ENABLED FALSE)
endif()

if(YOLO_ONNX_ROS_ENABLE_TENSORRT AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64" AND
     (EXISTS "/etc/nv_tegra_release" OR EXISTS "/usr/lib/aarch64-linux-gnu/tegra"))
    set(CMAKE_CUDA_ARCHITECTURES "72;87")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
  endif()
endif()

if(${PROJECT_NAME}_CUDA_ENABLED OR YOLO_ONNX_ROS_ENABLE_TENSORRT)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(${PROJECT_NAME}_CUDA_CATKIN_DEPENDS "CUDAToolkit")
  set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
  set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "CUDA::cudart")
endif()

configure_file(include/${PROJECT_NAME}/config.hpp.in ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}/config.hpp)

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION} ${${PROJECT_NAME}_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${${PROJECT_NAME}_CATKIN_DEPENDS}
  DEPENDS ${${PROJECT_NAME}_CUDA_CATKIN_DEPENDS} OpenCV console_bridge
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  SYSTEM
  ${${PROJECT_NAME}_CUDA_INCLUDE_DIRS}
  ${console_bridge_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/detection.cpp
  src/yolo_inference.cpp
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES} ${console_bridge_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
if(YOLO_ONNX_ROS_ENABLE_TENSORRT)
  find_package(Threads REQUIRED)
  target_sources(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_YOLOS_TRT_CUDA_SOURCE})
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
    ${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}
    ${tensorrt_ros_INCLUDE_DIRS}
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE ${tensorrt_ros_LIBRARIES} Threads::Threads)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()

add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

file(
  DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo11m/yolo11m.onnx ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo11m/yolo11m.onnx
  EXPECTED_HASH "SHA256=0eb68248f3868dfd8341e2e7d90e1e9d9594325e984574c11b0404f27673a419"
  SHOW_PROGRESS
)
file(
  DOWNLOAD https://github.com/tue-robotics/${PROJECT_NAME}/releases/download/yolo11m/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/resources/yolo11m/coco.yaml
  EXPECTED_HASH "SHA256=721773c881cb7de92339c9e0f86dbf6e90b9ed551c1cf8d5bd6f043497e241aa"
  SHOW_PROGRESS
)

# ------------------------------------------------------------------------------------------------
#                                               INSTALL
# ------------------------------------------------------------------------------------------------

install(FILES
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_INCLUDE_DESTINATION}/config.hpp
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

if(YOLO_ONNX_ROS_ENABLE_TENSORRT)
  install(
    DIRECTORY ${${PROJECT_NAME}_YOLOS_TRT_INCLUDE_DIR}/yolos
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  )
  install(
    FILES ${${PROJECT_NAME}_YOLOS_TRT_ROOT}/LICENSE
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/3rdparty/YOLOs-CPP-TensorRT
  )
endif()

install(
  TARGETS
    ${PROJECT_NAME}
    ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resources
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources
)

# ------------------------------------------------------------------------------------------------
#                                           Testing
# ------------------------------------------------------------------------------------------------
if (CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test("-W2 --ignore EXTERNAL_DIRECTORY --ignore EXTERNAL_FILE --ignore EXTERNAL_INCLUDE_PATH --ignore HEADER_OUTSIDE_PACKAGE_INCLUDE_PATH --ignore MISSING_DIRECTORY")

  # Utils unit tests (no models needed)
  catkin_add_gmock(yolo_test test/yolo_test.cpp WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/resources/yolo11m")
  if(TARGET yolo_test)
    target_link_libraries(
      yolo_test
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${OpenCV_LIBRARIES}
      )
  endif()

endif()
