#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <visualization_msgs/Marker.h>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <octomap/octomap.h>
#include <Eigen/Dense>

class OctreeUtils {
public:
    static visualization_msgs::Marker extractBoundingBox(const octomap::OcTree &octree,
                                                         const octomap::point3d &point,
                                                         const cv::Rect &yolobbox,
                                                         const Eigen::Affine3f &camToMap,
                                                         const Eigen::Matrix3f &ir_intrinsics);
};
