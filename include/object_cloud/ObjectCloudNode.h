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
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <std_srvs/Empty.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_map>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>
#include <image_transport/publisher.h>
#include <image_transport/image_transport.h>
#include <utility>
#include <vector>
#include <string>

#define VISUALIZE 1
#define VISUALIZE_OCTREE 1

static void prepareDepthData(const sensor_msgs::Image::ConstPtr& depth_msg, cv_bridge::CvImagePtr depth)
{
  depth->image = cv::Mat(depth_msg->height, depth_msg->width, CV_16UC1);
  // Taken from depth_image_proc
  // https://github.com/ros-perception/image_pipeline/blob/melodic/depth_image_proc/src/nodelets/convert_metric.cpp
  if (depth_msg->encoding == "32FC1")
  {
    auto cv_ptr = cv_bridge::toCvShare(depth_msg);
    uint16_t bad_point = 0;
    const auto* raw_data = reinterpret_cast<const float*>(&depth_msg->data[0]);
    auto* depth_data = reinterpret_cast<uint16_t*>(&depth->image.data[0]);
    for (unsigned index = 0; index < depth_msg->height * depth_msg->width; ++index)
    {
      float raw = raw_data[index];
      depth_data[index] = std::isnan(raw) ? bad_point : (uint16_t)(raw * 1000);
    }
  }
  else
  {
    memcpy(depth->image.data, depth_msg->data.data(), depth_msg->data.size());
  }
}

struct PointCloudRequest
{
  Eigen::Affine3f cam_to_target;
  Eigen::Vector2f xbounds;
  Eigen::Vector2f ybounds;
  Eigen::Vector2f zbounds;

  std::mutex mutex;
  std::condition_variable cond_var;
  pcl::PointCloud<pcl::PointXYZ>::Ptr result = nullptr;
};

class ObjectCloudNode
{
protected:
  Eigen::Matrix3f camera_intrinsics;

  // ROS tf2
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;

  // ObjectCloud backend
  ObjectCloud object_cloud;

  // Locks
  std::mutex global_mutex;

  // Octomap
  octomap::OcTree octree;

  // Objects data
  std::unordered_map<int, Object> entity_id_to_object;
  std::unordered_map<int, visualization_msgs::Marker> bounding_boxes;
  knowledge_rep::LongTermMemoryConduit ltmc;  // Knowledge base

  // Point cloud requests
  BlockingQueue<std::shared_ptr<PointCloudRequest>> point_cloud_requests;

  // Detections Publishers per camera
  std::unordered_map<std::string, ros::Publisher> detections_pubs;

  ros::NodeHandle node;

#ifdef VISUALIZE
  image_transport::Publisher viz_detections_pub;
  ros::Publisher viz_pub;
  ros::Publisher bbox_pub;
  ros::Publisher cloud_pub;
  ros::Publisher surfacecloud_pub;
  ros::Publisher surface_marker_pub;
#endif

  ros::ServiceServer clear_octree_server;
  ros::ServiceServer get_entities_server;
  ros::ServiceServer get_bounding_boxes_server;
  ros::ServiceServer get_objects_server;
  ros::ServiceServer get_surface_server;
  ros::ServiceServer surface_occupancy_server;

  void visualize();

  void updateBoundingBoxes(std::vector<std::pair<ImageBoundingBox, Object>> detection_objects,
                           const Eigen::Affine3f& cam_to_map);

  void processPointCloudRequests(cv_bridge::CvImagePtr depth);

  void runDetector(cv_bridge::CvImageConstPtr rgb_image, std::vector<ImageBoundingBox>& bboxes);

public:
  image_transport::ImageTransport it;
  bool received_first_message = false;

  explicit ObjectCloudNode(ros::NodeHandle node, const Eigen::Matrix3f& camera_intrinsics);

  ~ObjectCloudNode();

  void advertiseServices();

  void addToLtmc(const Object& object);

  bool clearOctree(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool getEntities(object_cloud::GetEntities::Request& req, object_cloud::GetEntities::Response& res);

  bool getObjects(object_cloud::GetObjects::Request& req, object_cloud::GetObjects::Response& res);

  bool getBoundingBoxes(object_cloud::GetBoundingBoxes::Request& req, object_cloud::GetBoundingBoxes::Response& res);

  bool getSurfaces(object_cloud::GetSurfaces::Request& req, object_cloud::GetSurfaces::Response& res);

  bool getSurfaceOccupancy(object_cloud::GetSurfaceOccupancy::Request& req,
                           object_cloud::GetSurfaceOccupancy::Response& res);
};
