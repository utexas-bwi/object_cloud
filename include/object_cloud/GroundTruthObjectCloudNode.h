#pragma once

#include <condition_variable>
#include <knowledge_representation/LongTermMemoryConduit.h>
#include <mutex>
#include <object_cloud/BlockingQueue.h>
#include <object_cloud/DetectedObject.h>
#include <object_cloud/GetBoundingBoxes.h>
#include <object_cloud/GetEntities.h>
#include <object_cloud/GetObjects.h>
#include <object_cloud/GetSurfaceOccupancy.h>
#include <object_cloud/GetSurfaces.h>
#include <object_cloud/ObjectCloud.h>
#include <object_cloud/ObjectCloudNode.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <std_srvs/Empty.h>
#include <unordered_map>

#define VISUALIZE 1
#define VISUALIZE_OCTREE 1

class GroundTruthObjectCloudNode : public ObjectCloudNode
{
public:
  explicit GroundTruthObjectCloudNode(ros::NodeHandle node, const Eigen::Matrix3f& camera_intrinsics);

  void dataCallback(const sensor_msgs::Image::ConstPtr& rgb_image, const sensor_msgs::Image::ConstPtr& depth_image,
                    const nav_msgs::Odometry::ConstPtr& odom);

  void handCameraCallback(const sensor_msgs::Image::ConstPtr& rgb_image);

private:
  ros::ServiceClient get_models_client;
  ros::ServiceClient get_properties_client;
  ros::ServiceClient get_state_client;
  ros::ServiceClient get_object_info_client;
};
