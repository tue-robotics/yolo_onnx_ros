cmake_minimum_required(VERSION 3.5)

project(yolo_onnx_ros)

add_compile_options(-Wall -Werror=all)
add_compile_options(-Wextra -Werror=extra)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED
  COMPONENTS
  onnxruntime_ros
)


set(${PROJECT_NAME}_CUDA_ENABLED ${onnxruntime_ros_CUDA_ENABLED})
if(${PROJECT_NAME}_CUDA_ENABLED)
  find_package(CUDAToolkit REQUIRED)
  set(${PROJECT_NAME}_CUDA_CATKIN_DEPENDS "CUDAToolkit")
  set(${PROJECT_NAME}_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
  set(${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES "CUDA::cudart")
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
  CATKIN_DEPENDS onnxruntime_ros
  DEPENDS ${${PROJECT_NAME}_CUDA_CATKIN_DEPENDS} OpenCV
)

# ------------------------------------------------------------------------------------------------
#                                           BUILD
# ------------------------------------------------------------------------------------------------

include_directories(
  include
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  SYSTEM
  ${${PROJECT_NAME}_CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/detection.cpp
  src/yolo_inference.cpp
)
target_link_libraries(${PROJECT_NAME} ${${PROJECT_NAME}_CUDA_TARGET_LINK_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
# add_dependencies(${PROJECT_NAME} generate_config_hpp)

add_executable(test_${PROJECT_NAME}
  src/main.cpp
)
target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

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
      )
  endif()

endif()
