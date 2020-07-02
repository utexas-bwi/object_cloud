#pragma once

#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

class PointCloudConstructor
{
public:
  static octomap::Pointcloud construct(const Eigen::Matrix3f& ir_intrinsics, const cv::Mat& depthImage,
                                       const Eigen::Affine3f& transform, float maxDist, const Eigen::Vector2f& xBounds,
                                       const Eigen::Vector2f& yBounds, const Eigen::Vector2f& zBounds);
};
