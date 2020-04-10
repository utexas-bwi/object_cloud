#pragma once

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <std_srvs/Empty.h>
#include <object_cloud/ObjectCloud.h>
#include <object_cloud/ObjectCloudNode.h>
#include <object_cloud/BlockingQueue.h>
#include <object_cloud/DetectedObject.h>
#include <object_cloud/GetEntities.h>
#include <object_cloud/GetBoundingBoxes.h>
#include <object_cloud/GetObjects.h>
#include <object_cloud/GetSurfaces.h>
#include <object_cloud/GetSurfaceOccupancy.h>
#include <knowledge_representation/LongTermMemoryConduit.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#define VISUALIZE 1
#define VISUALIZE_OCTREE 1

class GroundTruthObjectCloudNode : public ObjectCloudNode {

public:

  explicit GroundTruthObjectCloudNode(ros::NodeHandle node);


  void data_callback(const sensor_msgs::Image::ConstPtr &rgb_image,
                     const sensor_msgs::Image::ConstPtr &depth_image,
                     const nav_msgs::Odometry::ConstPtr &odom);

  void hand_camera_callback(const sensor_msgs::Image::ConstPtr &rgb_image);

private:
  ros::ServiceClient get_models_client;
  ros::ServiceClient get_properties_client;
  ros::ServiceClient get_state_client;
  ros::ServiceClient get_object_info_client;
};
