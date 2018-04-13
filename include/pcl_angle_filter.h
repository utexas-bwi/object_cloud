#ifndef VILLA_SURFACE_DETECTORS_PCL_ANGLE_FILTER_H
#define VILLA_SURFACE_DETECTORS_PCL_ANGLE_FILTER_H

#include "ros/ros.h"
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl/impl/point_types.hpp>
#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void filter_points_between(PointCloudT::ConstPtr input_cloud, PointCloudT::Ptr output_cloud, Eigen::Vector2f origin, float start_angle, float end_angle) {
    ROS_INFO("Removing input cloud points between %f and %f", start_angle, end_angle);


    ROS_INFO("Started with %zu points", input_cloud->size());

    float half_ignore_size = (end_angle - start_angle) / 2.0f;
    float mid_angle = (start_angle + end_angle) / 2.0f;
    Eigen::Vector2f ignore_range_bisect(cos(mid_angle), sin(mid_angle));
    ignore_range_bisect.normalize();

    for (PointCloudT::const_iterator i = input_cloud->begin(); i < input_cloud->end(); ++i) {
        PointT point = *i;
        Eigen::Vector2f point_vec(point.x, point.y);
        Eigen::Vector2f origin_to_center = point_vec - origin;
        origin_to_center.normalize();
        float point_cosine = origin_to_center.dot(ignore_range_bisect);
        if (acos(point_cosine) < half_ignore_size) {
            continue;
        }
        output_cloud->push_back(point);
    }
    ROS_INFO("Finished with %zu points", output_cloud->size());
}

#endif //VILLA_SURFACE_DETECTORS_PCL_ANGLE_FILTER_H
