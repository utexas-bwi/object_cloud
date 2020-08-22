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
#include <opencv2/features2d.hpp>
#include <vector>

#define VISUALIZE 1
#define VISUALIZE_OCTREE 1

class ColorBlobCloudNode : public ObjectCloudNode
{
protected:
  cv::Ptr<cv::SimpleBlobDetector> detector;

  void runDetector(cv_bridge::CvImageConstPtr rgb_image, std::vector<ImageBoundingBox>& bboxes);

public:
  explicit ColorBlobCloudNode(ros::NodeHandle node, const Eigen::Matrix3f& camera_intrinsics);

  void dataCallback(const sensor_msgs::Image::ConstPtr& rgb_msg, const sensor_msgs::Image::ConstPtr& depth_msg,
                    const nav_msgs::Odometry::ConstPtr& odom);
};
