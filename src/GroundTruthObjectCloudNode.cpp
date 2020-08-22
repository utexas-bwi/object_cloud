#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>
#include <object_msgs/ObjectInfo.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>

#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>

#include <object_cloud/GroundTruthObjectCloudNode.h>
#include <object_cloud/PointCloudConstructor.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv/highgui.h>

#include "object_cloud/ObjectCloud.h"
#include <gazebo_msgs/GetModelProperties.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/GetWorldProperties.h>

#include <utility>
#include <limits>
#include <string>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

GroundTruthObjectCloudNode::GroundTruthObjectCloudNode(ros::NodeHandle node, const Eigen::Matrix3f& camera_intrinsics)
  : ObjectCloudNode(node, camera_intrinsics)
{
  get_models_client = node.serviceClient<gazebo_msgs::GetWorldProperties>("/gazebo/get_world_properties");
  get_properties_client = node.serviceClient<gazebo_msgs::GetModelProperties>("/gazebo/get_model_properties");
  get_state_client = node.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
  get_object_info_client = node.serviceClient<object_msgs::ObjectInfo>("/gazebo_objects/get_info");

  get_models_client.waitForExistence();
  get_properties_client.waitForExistence();
  get_state_client.waitForExistence();
  get_object_info_client.waitForExistence();
}

void GroundTruthObjectCloudNode::dataCallback(const sensor_msgs::Image::ConstPtr& rgb_image,
                                              const sensor_msgs::Image::ConstPtr& depth_image,
                                              const nav_msgs::Odometry::ConstPtr& odom)
{
  std::lock_guard<std::mutex> global_lock(global_mutex);

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();

  received_first_message = true;

  // Beware! head_rgbd_sensor_rgb_frame is different from head_rgbd_sensor_link
  geometry_msgs::TransformStamped camToMapTransform;
  try
  {
    camToMapTransform =
        tf_buffer.lookupTransform("map", rgb_image->header.frame_id, rgb_image->header.stamp, ros::Duration(0.02));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return;
  }

  Eigen::Affine3f camToMap = tf2::transformToEigen(camToMapTransform).cast<float>();

  cv::Mat depthI(depth_image->height, depth_image->width, CV_16UC1);
  memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());

  // Process point cloud requests

  float inf = std::numeric_limits<float>::infinity();
  while (!point_cloud_requests.empty())
  {
    std::shared_ptr<PointCloudRequest> req = point_cloud_requests.pop();
    std::cout << "PROCESSING" << std::endl;
    {
      std::lock_guard<std::mutex> lock(req->mutex);

      octomap::Pointcloud planecloud = PointCloudConstructor::construct(camera_intrinsics, depthI, req->cam_to_target,
                                                                        inf, req->xbounds, req->ybounds, req->zbounds);
      pcl::PointCloud<pcl::PointXYZ>::Ptr req_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      req_cloud->points.reserve(planecloud.size());
      for (const auto& p : planecloud)
      {
        req_cloud->points.emplace_back(p.x(), p.y(), p.z());
      }
      req->result = req_cloud;
    }
    std::cout << "PROCESSED" << std::endl;

    req->cond_var.notify_one();
    std::cout << "NOTIFIED" << std::endl;
  }

  gazebo_msgs::GetWorldProperties all_models_srv;
  if (!get_models_client.call(all_models_srv))
  {
    ROS_ERROR("Failed to call service gazebo/get_world_properties");
    return;
  }
  // ROS_INFO_STREAM(all_models_srv.response);

  for (std::string& name : all_models_srv.response.model_names)
  {
    gazebo_msgs::GetModelProperties model_properties_srv;
    model_properties_srv.request.model_name = name;
    get_properties_client.call(model_properties_srv);

    if (!model_properties_srv.response.is_static && model_properties_srv.response.geom_names.size() == 1)
    {
      gazebo_msgs::GetModelState model_state_srv;
      model_state_srv.request.model_name = name;

      // ROS_INFO_STREAM(name);
      get_state_client.call(model_state_srv);

      auto position = model_state_srv.response.pose.position;

      Object object;
      object.position = Eigen::Vector3f(position.x, position.y, position.z);
      object.label = name;
      std::pair<bool, Object> ret = object_cloud.addObject(object);

      // If object was added to object_cloud, add to knowledge base
      bool newObj = ret.first;
      if (newObj)
      {
        std::cout << "New Object " << ret.second.position << std::endl;
        addToLtmc(ret.second);
      }

      // Bounding boxes
      object_msgs::ObjectInfo object_info_srv;
      object_info_srv.request.name = name;
      object_info_srv.request.get_geometry = true;
      // ROS_INFO_STREAM(object_info_srv.request);
      get_object_info_client.call(object_info_srv);
      // ROS_INFO_STREAM(object_info_srv.response);
      auto dimensions = object_info_srv.response.object.primitives[0].dimensions;
      auto pose = object_info_srv.response.object.origin;

      visualization_msgs::Marker marker;
      marker.header.frame_id = "map";
      marker.id = 1;
      marker.type = 1;
      marker.action = 0;
      marker.pose = pose;
      marker.scale.x = dimensions[0];
      marker.scale.y = dimensions[1];
      marker.scale.z = dimensions[2];
      marker.color.r = 0.;
      marker.color.b = 0.;
      marker.color.g = 1.;
      marker.color.a = 1.;
      marker.lifetime = ros::Duration(0);

      int key = ret.second.id;
      auto mit = bounding_boxes.find(key);
      if (mit != bounding_boxes.end())
      {
        bounding_boxes.at(key) = marker;
      }
      else
      {
        bounding_boxes.insert({ key, marker });
      }
    }
  }

  // Parts of the depth image that have objects
  // This will be useful for constructing a region-of-interest Point Cloud
  cv::Mat depthMasked = cv::Mat::zeros(depth_image->height, depth_image->width, CV_16UC1);
  // If the robot is moving then don't update Octomap
  if (!(Eigen::Vector3f(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z).norm() >
            0.05 ||
        Eigen::Vector3f(odom->twist.twist.angular.x, odom->twist.twist.angular.y, odom->twist.twist.angular.z).norm() >
            0.05))
  {
    // Use depthMasked to construct a ROI Point Cloud for use with Octomap
    // Without this, Octomap takes forever
    Eigen::Vector2f nobounds(-inf, inf);
    octomap::Pointcloud cloud = PointCloudConstructor::construct(camera_intrinsics, depthMasked, camToMap, 3., nobounds,
                                                                 nobounds, Eigen::Vector2f(0., inf));

    // Insert ROI PointCloud into Octree
    Eigen::Vector3f origin = camToMap * Eigen::Vector3f::Zero();
    octree.insertPointCloud(cloud, octomap::point3d(origin(0), origin(1), origin(2)),
                            3,      // Max range of 3. This isn't meters, I don't know wtf this is.
                            false,  // We don't want lazy updates
                            true);  // Discretize speeds it up by approximating
  }

#if (VISUALIZE)

// This takes time so disable if not debugging...
#if (VISUALIZE_OCTREE)
  // Publish Octree
  octomap_msgs::Octomap cloudMsg;
  cloudMsg.header.frame_id = "map";
  octomap_msgs::binaryMapToMsg(octree, cloudMsg);
  cloud_pub.publish(cloudMsg);
#endif

  // Publish bounding boxes
  visualization_msgs::MarkerArray boxes;
  int id = 0;
  for (const auto& e : bounding_boxes)
  {
    visualization_msgs::Marker box = e.second;
    box.id = id;
    boxes.markers.push_back(box);
    id++;
  }
  bbox_pub.publish(boxes);

  // Publish cloud
  visualize();
#endif

  // Record end time
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  // std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}

// TODO(yuqianjiang)
void GroundTruthObjectCloudNode::handCameraCallback(const sensor_msgs::Image::ConstPtr& rgb_image)
{
}
