cmake_minimum_required(VERSION 2.8.3)
project(villa_surface_detectors)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
        rospy
	pcl_ros
	pcl_conversions
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
	std_srvs
 	message_generation
		villa_manipulation
	tmc_yolo2_ros
  villa_octomap_server
)

find_package(Eigen3 REQUIRED)

#add_service_files(
#   FILES
#)

add_action_files(
   DIRECTORY action
   FILES PerceiveShelf.action
   PerceiveTable.action
)

generate_messages(
   DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    std_srvs
    actionlib_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS actionlib_msgs
  DEPENDS message_runtime
#  	  bwi_tabletop_perception
	  eigen
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)

add_executable(shelf_detector_action_server src/shelf_detector_action_server.cpp src/segment_objects.cpp src/detect_horizontal_planes.cpp src/BoundingBox.cpp src/VillaSurfaceDetector.cpp)
add_dependencies(shelf_detector_action_server villa_surface_detectors_generate_messages_cpp)
target_link_libraries(shelf_detector_action_server ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(shelf_detector_action_tester src/shelf_detector_action_tester.cpp)
add_dependencies(shelf_detector_action_tester villa_surface_detectors_generate_messages_cpp)
target_link_libraries(shelf_detector_action_tester ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(table_detector_action_server src/table_detector_action_server.cpp src/segment_objects.cpp src/detect_horizontal_planes.cpp src/BoundingBox.cpp src/VillaSurfaceDetector.cpp)
add_dependencies(table_detector_action_server villa_surface_detectors_generate_messages_cpp)
target_link_libraries(table_detector_action_server ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# TODO: Address build issues
# See issue #6
#add_executable(table_detector_action_tester src/table_detector_action_tester.cpp)
#add_dependencies(table_detector_action_tester villa_surface_detectors_generate_messages_cpp)
#target_link_libraries(table_detector_action_tester ${catkin_LIBRARIES} ${PCL_LIBRARIES})


install(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
