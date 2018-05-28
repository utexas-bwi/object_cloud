cmake_minimum_required(VERSION 3.0)
project(villa_yolocloud)

add_compile_options(-std=c++11)

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
    tf_conversions
    tmc_yolo2_ros
    knowledge_representation
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

add_message_files(
    FILES
    DetectedObject.msg
    DetectedObjectArray.msg
)

add_service_files(
    FILES
    GetEntities.srv
    GetShelfObjects.srv
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
    eigen
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

add_executable(yolocloud_node src/yolocloud_node.cpp src/YoloCloud.cpp)
add_dependencies(yolocloud_node ${villa_yolocloud_EXPORTED_TARGETS})
target_link_libraries(yolocloud_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

install(PROGRAMS
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h")

