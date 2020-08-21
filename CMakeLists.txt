cmake_minimum_required(VERSION 3.0)
project(object_cloud)

if(NOT CMAKE_BUILD_TYPE)
 # set(CMAKE_BUILD_TYPE Release)
 # set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG")
 # set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
endif()
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
    cv_bridge
    geometry_msgs
    image_transport
    knowledge_representation
    message_generation
    message_filters
    message_runtime
    octomap_ros
    pcl_ros
    pcl_conversions
    roscpp
    roslint
    sensor_msgs
    std_msgs
    std_srvs
    tf2
    tf2_eigen
    visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)

add_message_files(
    FILES
    DetectedObject.msg
    SearchBox.msg
    BoundingBox2D.msg
    BoundingBox2DList.msg
)

add_service_files(
    FILES
    GetEntities.srv
    GetBoundingBoxes.srv
    GetObjects.srv
    GetSurfaces.srv
    GetSurfaceOccupancy.srv
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
    DEPENDS
    CATKIN_DEPENDS geometry_msgs image_transport knowledge_representation message_runtime octomap_ros pcl_ros sensor_msgs std_msgs std_srvs visualization_msgs
    LIBRARIES object_cloud
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)

# Add compile definitions this way is a recent CMake feature...
if(${CMAKE_VERSION} VERSION_GREATER "3.12.0")
  # Stops a debug output that fires every time the octree is serialized
  add_compile_definitions(OCTOMAP_NODEBUGOUT)
endif()

add_library(object_cloud src/ObjectCloud.cpp src/ObjectCloudNode.cpp src/PointCloudConstructor.cpp src/PointCloudUtils.cpp)
add_dependencies(object_cloud ${object_cloud_EXPORTED_TARGETS})
target_link_libraries(object_cloud         ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OCTOMAP_LIBRARIES})

add_executable(color_blob_cloud_node src/color_blob_cloud_node.cpp src/ColorBlobCloudNode.cpp)
add_dependencies(color_blob_cloud_node ${object_cloud_EXPORTED_TARGETS})
target_link_libraries(color_blob_cloud_node
        object_cloud
        )

find_package(object_msgs)
if (object_msgs_FOUND)
  include_directories(${object_msgs_INCLUDE_DIRS})
  add_executable(ground_truth_cloud_node src/ground_truth_cloud_node.cpp src/GroundTruthObjectCloudNode.cpp)
  add_dependencies(ground_truth_cloud_node ${object_cloud_EXPORTED_TARGETS})
  target_link_libraries(ground_truth_cloud_node
    object_cloud
  )
endif()

### INSTALL

install(PROGRAMS
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS object_cloud color_blob_cloud_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

### LINT

file(GLOB_RECURSE ${PROJECT_NAME}_CPP_SRC
        RELATIVE ${PROJECT_SOURCE_DIR} src/*.cpp  include/${PROJECT_NAME}/*.h test/*.cpp)
set(ROSLINT_CPP_OPTS "--filter=-legal/copyright,-build/header_guard,-runtime/references,-build/c++11,-whitespace/braces")
roslint_cpp(${${PROJECT_NAME}_CPP_SRC})


set(ROSLINT_PYTHON_OPTS "--max-line-length=120")
roslint_python(${${PROJECT_NAME}_PY_SCRIPTS})
roslint_python(${${PROJECT_NAME}_PY_SRC})
