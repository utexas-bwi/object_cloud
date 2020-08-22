#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <boost/optional.hpp>
#include <iostream>
#include <octomap/octomap.h>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <string>
#include <visualization_msgs/Marker.h>
#include <vector>

class PointCloudUtils
{
public:
  static visualization_msgs::Marker extractBoundingBox(const octomap::OcTree& octree, const octomap::point3d& point,
                                                       const cv::Rect& bbox, const Eigen::Affine3f& cam_to_map,
                                                       const Eigen::Affine3f& base_to_map,
                                                       const Eigen::Matrix3f& depth_intrinsics);

  static boost::optional<visualization_msgs::Marker> extractSurface(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    const Eigen::Affine3f& camToMap,
                                                                    const Eigen::Affine3f& camToStraightCam,
                                                                    float xScale, float yScale, float zScale,
                                                                    ros::Publisher& publisher);

  static std::vector<Eigen::Vector2f> surfaceOccupancy(std::vector<visualization_msgs::Marker> all_objects,
                                                       visualization_msgs::Marker surface, Eigen::Vector2f xbounds,
                                                       Eigen::Vector2f ybounds);
};
