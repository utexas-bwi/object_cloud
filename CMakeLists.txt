cmake_minimum_required(VERSION 3.0)
project(villa_yolocloud)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
add_compile_options(-std=c++11)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(DEFINED ENV{GPU_ENABLED})
  set(USE_GPU true)
else()
  set(USE_GPU false)
endif()

if(DEFINED ENV{CUDNN_ENABLED})
  set(USE_CUDNN true)
else()
  set(USE_CUDNN false)
endif()

if(USE_GPU)
  add_definitions(-DGPU)
  find_package(CUDA REQUIRED)
  if(USE_CUDNN)
    add_definitions(-DCUDNN)
  endif()
endif()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    pcl_ros
    pcl_conversions
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    std_srvs
    message_generation
    message_filters
    cv_bridge
    tf2
    tf2_eigen
    image_transport
    octomap_ros
    knowledge_representation
    villa_darknet
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)

add_message_files(
    FILES
    DetectedObject.msg
    DetectedObjectArray.msg
)

add_service_files(
    FILES
    GetEntities.srv
    GetBoundingBoxes.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    std_srvs
)

catkin_package(
    INCLUDE_DIRS include
    DEPENDS message_runtime
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)

if(USE_GPU)
  include_directories(
    SYSTEM ${CUDA_INCLUDE_DIRS}
    /usr/local/cuda/targets/x86_64-linux/include
  )
endif()

add_executable(yolocloud_node src/yolocloud_node.cpp src/YoloCloud.cpp src/YoloModel.cpp src/PointCloudConstructor.cpp include/villa_yolocloud/PointCloudConstructor.h)
add_dependencies(yolocloud_node ${villa_yolocloud_EXPORTED_TARGETS})
target_link_libraries(yolocloud_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OCTOMAP_LIBRARIES}
)

install(PROGRAMS
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h")

