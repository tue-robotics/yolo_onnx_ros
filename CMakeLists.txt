cmake_minimum_required(VERSION 3.5)
project(yolo_onnx_ros)

add_compile_options(-Wall -Werror=all)
add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(catkin REQUIRED
  COMPONENTS
  onnxruntime_ros
)


set(${PROJECT_NAME}_CUDA_ENABLED ${onnxruntime_ros_CUDA_ENABLED})
if(onnxruntime_ros_CUDA_ENABLED)
  find_package(CUDAToolkit REQUIRED)
endif()

configure_file(include/${PROJECT_NAME}/config.hpp.in ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}/config.hpp)
# add_custom_target(generate_config_hpp
#   DEPENDS ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}/config.hpp
# )

# ------------------------------------------------------------------------------------------------
#                                        CATKIN EXPORT
# ------------------------------------------------------------------------------------------------

catkin_package(
  INCLUDE_DIRS include ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS CUDAToolkit OpenCV
)


# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  SYSTEM
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

# -------------- ONNXRuntime  ------------------#

add_library(${PROJECT_NAME}
  src/yolo_inference.cpp
)
# add_dependencies(${PROJECT_NAME} generate_config_hpp)
target_link_libraries(${PROJECT_NAME} CUDA::toolkit ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(test_${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

# For Windows system, copy onnxruntime.dll to the same folder of the executable file
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()


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

install(
  TARGETS
    ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
  TARGETS
    ${PROJECT_NAME}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Download https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml
# and put it in the same folder of the executable file
configure_file(data/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)

# Copy yolov8n.onnx file to the same folder of the executable file
# configure_file(model/yolo11m.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolo11m.onnx COPYONLY)

# # Create folder name images in the same folder of the executable file
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
# )
