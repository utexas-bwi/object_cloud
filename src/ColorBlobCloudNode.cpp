#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/Image.h>

#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <object_cloud/ColorBlobCloudNode.h>
#include <object_cloud/PointCloudConstructor.h>

#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv2/features2d.hpp>
#include <limits>
#include <utility>
#include <algorithm>
#include <vector>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using std::pair;
using std::vector;

inline bool is_moving(const nav_msgs::Odometry::ConstPtr& odom)
{
  return Eigen::Vector3f(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z).norm() >
             0.05 ||
         Eigen::Vector3f(odom->twist.twist.angular.x, odom->twist.twist.angular.y, odom->twist.twist.angular.z).norm() >
             0.05;
}

ColorBlobCloudNode::ColorBlobCloudNode(ros::NodeHandle node, const Eigen::Matrix3f& camera_intrinsics)
  : ObjectCloudNode(node, camera_intrinsics)
{
  auto params = cv::SimpleBlobDetector::Params();
  params.minThreshold = 10;
  params.maxThreshold = 200;

  params.filterByArea = true;
  params.minArea = 1500;
  params.maxArea = 5000;

  params.filterByCircularity = true;
  params.minCircularity = 0.1;

  params.filterByConvexity = true;
  params.minConvexity = 0.87;

  params.filterByInertia = false;
  params.minInertiaRatio = 0.01;
  detector = cv::SimpleBlobDetector::create(params);
}

void ColorBlobCloudNode::runDetector(cv_bridge::CvImageConstPtr rgb_image, vector<ImageBoundingBox>& bboxes)
{
  std::vector<cv::KeyPoint> keypoints;
  // cv::Mat red;
  // cv::extractChannel(rgb_image->image, red, 2);
  detector->detect(rgb_image->image, keypoints);
  /*
  cv::Mat output;
  drawKeypoints( red, keypoints, output, cv::Scalar(255,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  imshow("keypoints", output );
  cv::waitKey(1);
  */

  // Convert the detections into our bbox format
  for (const auto& keypoint : keypoints)
  {
    // Keypoints are center point + diameter
    ImageBoundingBox bbox;
    const auto radius = keypoint.size / 2;
    bbox.x = keypoint.pt.x - radius;
    bbox.y = keypoint.pt.y - radius;
    bbox.width = keypoint.size;
    bbox.height = keypoint.size;
    bbox.label = std::to_string(keypoint.class_id);
    bboxes.push_back(bbox);
  }
}

void ColorBlobCloudNode::dataCallback(const sensor_msgs::Image::ConstPtr& rgb_msg,
                                      const sensor_msgs::Image::ConstPtr& depth_msg,
                                      const nav_msgs::Odometry::ConstPtr& odom)
{
  std::lock_guard<std::mutex> global_lock(global_mutex);

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();

  received_first_message = true;

  cv_bridge::CvImageConstPtr rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
  vector<ImageBoundingBox> bboxes;
  runDetector(rgb, bboxes);

  geometry_msgs::TransformStamped cam_to_map_transform;
  geometry_msgs::TransformStamped base_to_map_transform;
  try
  {
    cam_to_map_transform =
        tf_buffer.lookupTransform("map", rgb_msg->header.frame_id, rgb_msg->header.stamp, ros::Duration(0.02));
    // base_to_map_transform = tf_buffer.lookupTransform("map", "base_link", rgb_image->header.stamp,
    //                                              ros::Duration(0.02));
    base_to_map_transform = cam_to_map_transform;
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return;
  }

  Eigen::Affine3f cam_to_map = tf2::transformToEigen(cam_to_map_transform).cast<float>();
  Eigen::Affine3f base_to_map = tf2::transformToEigen(base_to_map_transform).cast<float>();

  cv_bridge::CvImagePtr depth(new cv_bridge::CvImage());
  prepareDepthData(depth_msg, depth);

  processPointCloudRequests(depth);

  // Parts of the depth image that have objects
  // This will be useful for constructing a region-of-interest Point Cloud
  cv::Mat depth_masked = cv::Mat::zeros(depth_msg->height, depth_msg->width, CV_16UC1);

  vector<pair<ImageBoundingBox, Object>> detection_objects;

  for (const auto& bbox : bboxes)
  {
    // If the bounding box is at the edge of the image, ignore it
    if (bbox.x == 0 || bbox.x + bbox.width >= rgb->image.cols || bbox.y == 0 || bbox.y + bbox.height >= rgb->image.rows)
    {
      continue;
    }

    // 10 pixel buffer
    int min_x = std::max(0., bbox.x - 10.);
    int min_y = std::max(0., bbox.y - 10.);
    int max_x = std::min(rgb->image.cols - 1., bbox.x + bbox.width + 20.);
    int max_y = std::min(rgb->image.rows - 1., bbox.y + bbox.height + 20.);
    cv::Rect region(min_x, min_y, max_x - min_x, max_y - min_y);
    depth->image(region).copyTo(depth_masked(region));

    std::pair<bool, Object> ret = object_cloud.addObject(bbox, rgb->image, depth->image, cam_to_map);
    if (!ret.second.invalid())
    {
      detection_objects.emplace_back(bbox, ret.second);
    }

    // If object was added, add to knowledge base
    bool newObj = ret.first;
    if (newObj)
    {
      // std::cout << "New Object " << ret.second.position << std::endl;
      addToLtmc(ret.second);
    }
  }

  // OCTOMAP UPDATE--------------------------

  // If the robot is moving then don't update Octomap
  if (!is_moving(odom))
  {
    // Use depthMasked to construct a ROI Point Cloud for use with Octomap
    // Without this, Octomap takes forever
    float inf = std::numeric_limits<float>::infinity();
    Eigen::Vector2f nobounds(-inf, inf);
    octomap::Pointcloud cloud = PointCloudConstructor::construct(camera_intrinsics, depth_masked, cam_to_map, 3.,
                                                                 nobounds, nobounds, Eigen::Vector2f(0., inf));
    if (cloud.size() != 0)
    {
      // Insert ROI PointCloud into Octree
      Eigen::Vector3f origin = cam_to_map * Eigen::Vector3f::Zero();
      octree.insertPointCloud(cloud, octomap::point3d(origin(0), origin(1), origin(2)), 3,
                              false,  // We don't want lazy updates
                              true);  // Discretize speeds it up by approximating

      updateBoundingBoxes(detection_objects, cam_to_map);
    }
  }
#if (VISUALIZE)
  cv::Mat rgb_copy = rgb->image;
  if (viz_detections_pub.getNumSubscribers() > 0)
  {
    for (const auto& d : detection_objects)
    {
      cv::rectangle(rgb_copy, d.first, cv::Scalar(0, 0, 255), 3);
    }
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb->image).toImageMsg();
    viz_detections_pub.publish(msg);
  }

#endif
  // Record end time
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}
