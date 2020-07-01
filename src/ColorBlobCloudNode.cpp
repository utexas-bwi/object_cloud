#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <object_msgs/ObjectInfo.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <knowledge_representation/LTMCInstance.h>
#include <knowledge_representation/convenience.h>
#include <object_cloud/BoundingBox2DList.h>
#include <object_cloud/ColorBlobCloudNode.h>
#include <object_cloud/PointCloudConstructor.h>
#include <object_cloud/PointCloudUtils.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv/highgui.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/features2d.hpp>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

ColorBlobCloudNode::ColorBlobCloudNode(ros::NodeHandle node)
    : ObjectCloudNode(node) {
  auto params = cv::SimpleBlobDetector::Params();
  params.minThreshold = 10;
  params.maxThreshold = 200;

  params.filterByArea = true;
  params.minArea = 1500;

  params.filterByCircularity = true;
  params.minCircularity = 0.1;

  params.filterByConvexity = true;
  params.minConvexity = 0.87;

  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;
  detector = cv::SimpleBlobDetector::create(params);

}

void ColorBlobCloudNode::data_callback(
    const sensor_msgs::Image::ConstPtr &rgb_image,
    const sensor_msgs::Image::ConstPtr &depth_image,
    const nav_msgs::Odometry::ConstPtr &odom) {
  std::lock_guard<std::mutex> global_lock(global_mutex);

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();

  received_first_message = true;

  std::vector<cv::KeyPoint> keypoints;
  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat test;

  cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
  detector->detect(cv_ptr->image, keypoints);

  drawKeypoints( cv_ptr->image, keypoints, test, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  imshow("keypoints", test );// Show blobs
  cv::waitKey(1);


  geometry_msgs::TransformStamped camToMapTransform;
  try {
    camToMapTransform =
        tf_buffer.lookupTransform("map", rgb_image->header.frame_id,
                                  rgb_image->header.stamp, ros::Duration(0.02));
  } catch (tf2::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return;
  }

  Eigen::Affine3f camToMap =
      tf2::transformToEigen(camToMapTransform).cast<float>();

  cv::Mat depthI(depth_image->height, depth_image->width, CV_16UC1);
  memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());

  // Process point cloud requests
  Eigen::Matrix3f ir_intrinsics;
  ir_intrinsics << 535.2900990271, 0, 320.0, 0, 535.2900990271, 240.0, 0, 0, 1;
  float inf = std::numeric_limits<float>::infinity();
  while (!point_cloud_requests.empty()) {
    std::shared_ptr<PointCloudRequest> req = point_cloud_requests.pop();
    std::cout << "PROCESSING" << std::endl;
    {
      std::lock_guard<std::mutex> lock(req->mutex);

      octomap::Pointcloud planecloud = PointCloudConstructor::construct(
          ir_intrinsics, depthI, req->cam_to_target, inf, req->xbounds,
          req->ybounds, req->zbounds);
      pcl::PointCloud<pcl::PointXYZ>::Ptr req_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      req_cloud->points.reserve(planecloud.size());
      for (const auto &p : planecloud) {
        req_cloud->points.emplace_back(p.x(), p.y(), p.z());
      }
      req->result = req_cloud;
    }
    std::cout << "PROCESSED" << std::endl;

    req->cond_var.notify_one();
    std::cout << "NOTIFIED" << std::endl;
  }





  // Parts of the depth image that have objects
  // This will be useful for constructing a region-of-interest Point Cloud
  cv::Mat depthMasked =
      cv::Mat::zeros(depth_image->height, depth_image->width, CV_16UC1);





  std::vector<std::pair<cv::KeyPoint, Object>> detectionPositions;

  for (const auto &detection : keypoints) {
    ImageBoundingBox bbox;
    bbox.x = detection.pt.x;
    bbox.y = detection.pt.y;
    bbox.width = detection.size;
    bbox.height = detection.size;
    bbox.label = std::to_string(detection.class_id);

    // If the bounding box is at the edge of the image, ignore it
    if (bbox.x == 0 || bbox.x + bbox.width >= cv_ptr->image.cols
        || bbox.y == 0 || bbox.y + bbox.height >= cv_ptr->image.rows) {
      continue;
    }

    // 10 pixel buffer
    int min_x = std::max(0., detection.pt.x - 10.);
    int min_y = std::max(0., detection.pt.y - 10.);
    int max_x = std::min(cv_ptr->image.cols - 1., detection.pt.x + detection.size + 20.);
    int max_y = std::min(cv_ptr->image.rows - 1., detection.pt.y + detection.size + 20.);
    cv::Rect region(min_x, min_y, max_x - min_x, max_y - min_y);
    depthI(region).copyTo(depthMasked(region));

    std::pair<bool, Object> ret = object_cloud.add_object(bbox, cv_ptr->image, depthI, camToMap);
    if (!ret.second.invalid()) {
      detectionPositions.emplace_back(detection, ret.second);
    }

    // If object was added to yolocloud, add to knowledge base
    bool newObj = ret.first;
    if (newObj) {
      std::cout << "New Object " << ret.second.position << std::endl;
      add_to_ltmc(ret.second);
    }
  }

      //OCTOMAP UPDATE_--------------------------

  // If the robot is moving then don't update Octomap
  if (!(Eigen::Vector3f(odom->twist.twist.linear.x, odom->twist.twist.linear.y,
                        odom->twist.twist.linear.z)
                .norm() > 0.05 ||
        Eigen::Vector3f(odom->twist.twist.angular.x,
                        odom->twist.twist.angular.y,
                        odom->twist.twist.angular.z)
                .norm() > 0.05)) {
    // Use depthMasked to construct a ROI Point Cloud for use with Octomap
    // Without this, Octomap takes forever
    Eigen::Vector2f nobounds(-inf, inf);
    octomap::Pointcloud cloud = PointCloudConstructor::construct(
        ir_intrinsics, depthMasked, camToMap, 3., nobounds, nobounds,
        Eigen::Vector2f(0., inf));

    // Insert ROI PointCloud into Octree
    Eigen::Vector3f origin = camToMap * Eigen::Vector3f::Zero();
    octree.insertPointCloud(
        cloud, octomap::point3d(origin(0), origin(1), origin(2)),
        3,     // Max range of 3. This isn't meters, I don't know wtf this is.
        false, // We don't want lazy updates
        true); // Discretize speeds it up by approximating
  }

  {
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
  for (const auto &e : bounding_boxes) {
    visualization_msgs::Marker box = e.second;
    box.id = id;
    boxes.markers.push_back(box);
    id++;
  }
  bbox_pub.publish(boxes);

  // Publish cloud
  visualize();
  #endif
  }

  // Record end time
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  // std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}

