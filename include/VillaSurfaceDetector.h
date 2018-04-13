#ifndef VILLA_SURFACE_DETECTOR_CLASS_H
#define VILLA_SURFACE_DETECTOR_CLASS_H

#include <vector>
#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BoundingBox.h"
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>

#define Z_AXIS_REFERENCE_FRAME "base_link" // Find planes perpendicular to the z-axis of this frame

/* define what kind of point clouds we're using */
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace std;

static bool compare_by_second(const pair<int, float> &lhs, const pair<int, float> &rhs) {
	return lhs.second < rhs.second;
}

class VillaSurfaceDetector {

private:
  tf::TransformListener *tf_listener_;

  visualization_msgs::Marker createBoundingBoxMarker(const BoundingBox &box, const int &marker_index, string frame);
  BoundingBox extract_aa_bounding_box_params(const PointCloudT::Ptr &plane_cloud);
  BoundingBox extract_oriented_bounding_box_params(const PointCloudT::Ptr &plane_cloud);
  std::vector<PointCloudT::Ptr> isolate_surfaces(const PointCloudT::Ptr &in, double tolerance);
  void move_to_frame(const PointCloudT::Ptr &input, const string &target_frame, PointCloudT::Ptr &output);
  double calculate_density(const PointCloudT::Ptr &cloud, const BoundingBox &box);
  void vector_to_frame(const geometry_msgs::Vector3Stamped &vector, const string &target_frame,
    geometry_msgs::Vector3Stamped &out_vector);


public:
  VillaSurfaceDetector() { }
  VillaSurfaceDetector(ros::NodeHandle &nh) {
    tf_listener_ = new tf::TransformListener(nh);
  }
  void init(ros::NodeHandle &nh) {
    tf_listener_ = new tf::TransformListener(nh);
  }

  bool detect_horizontal_planes(Eigen::Vector3f axis, const PointCloudT &cloud_input,
    vector<PointCloudT> &horizontal_planes_out,
    vector<geometry_msgs::Quaternion> &horizontal_plane_coefs_out,
    vector<visualization_msgs::Marker> &horizontal_plane_bounding_boxes_out,
    vector<visualization_msgs::Marker> &horizontal_plane_AA_bounding_boxes_out);

  bool segment_objects(Eigen::Vector3f axis, const PointCloudT::Ptr &world_points,
    const geometry_msgs::Quaternion &plane_coefs,
    const visualization_msgs::Marker &plane_bounding_box,
    float maximum_height,
    vector<PointCloudT::Ptr> &objects_out,
    vector<visualization_msgs::Marker> &object_bounding_boxes_out);

	void pressEnter(string message);
};

#endif //VILLA_SURFACE_DETECTOR_CLASS_H
