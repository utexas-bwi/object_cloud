#pragma once

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <std_srvs/Empty.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <villa_object_cloud/ObjectCloud.h>
#include <villa_object_cloud/YoloModel.h>
#include <villa_object_cloud/BlockingQueue.h>
#include <villa_object_cloud/DetectedObject.h>
#include <villa_object_cloud/GetEntities.h>
#include <villa_object_cloud/GetBoundingBoxes.h>
#include <villa_object_cloud/GetObjects.h>
#include <villa_object_cloud/GetSurfaces.h>
#include <villa_object_cloud/GetSurfaceOccupancy.h>
#include <knowledge_representation/LongTermMemoryConduit.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#define VISUALIZE 1
#define VISUALIZE_OCTREE 1

struct PointCloudRequest {
  Eigen::Affine3f cam_to_target;
  Eigen::Vector2f xbounds;
  Eigen::Vector2f ybounds;
  Eigen::Vector2f zbounds;

  std::mutex mutex;
  std::condition_variable cond_var;
  pcl::PointCloud<pcl::PointXYZ>::Ptr result = nullptr;
};

class ObjectCloudNode {
protected:
  // ROS tf2
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;

  // ObjectCloud backend
  ObjectCloud object_cloud;

  // Locks
  std::mutex global_mutex;  // Global lock

  // Octomap
  octomap::OcTree octree;

  // Objects data
  std::unordered_map<int, Object> entity_id_to_object;
  std::unordered_map<int, visualization_msgs::Marker> bounding_boxes;
  knowledge_rep::LongTermMemoryConduit ltmc; // Knowledge base

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
  ros::Publisher pose_pub;
#endif

  ros::ServiceServer clear_octree_server;
  ros::ServiceServer get_entities_server;
  ros::ServiceServer get_bounding_boxes_server;
  ros::ServiceServer get_objects_server;
  ros::ServiceServer get_surface_server;
  ros::ServiceServer surface_occupancy_server;

  void visualize();

public:
  image_transport::ImageTransport it;
  bool received_first_message = false;

  explicit ObjectCloudNode(ros::NodeHandle node);

  ~ObjectCloudNode();

  void advertise_services();

  void add_to_ltmc(const Object &object);

  bool clear_octree(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);

  bool get_entities(villa_object_cloud::GetEntities::Request &req, villa_object_cloud::GetEntities::Response &res);

  bool get_objects(villa_object_cloud::GetObjects::Request &req, villa_object_cloud::GetObjects::Response &res);

  bool get_bounding_boxes(villa_object_cloud::GetBoundingBoxes::Request &req,
                          villa_object_cloud::GetBoundingBoxes::Response &res);

  bool get_surfaces(villa_object_cloud::GetSurfaces::Request &req,
                    villa_object_cloud::GetSurfaces::Response &res);

  bool get_surface_occupancy(villa_object_cloud::GetSurfaceOccupancy::Request &req,
                             villa_object_cloud::GetSurfaceOccupancy::Response &res);
};
