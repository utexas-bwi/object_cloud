#pragma once

#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <octomap/octomap.h>
#include <octomap/Pointcloud.h>

#include <Eigen/Dense>

class PointCloudConstructor {
public:
    static octomap::Pointcloud construct(const Eigen::Matrix3f &ir_intrinsics,
                                         const cv::Mat &depthImage,
                                         const Eigen::Affine3f &transform);
};
