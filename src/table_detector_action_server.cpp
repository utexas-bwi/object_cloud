#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>

// PCL specific includes
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/point_cloud.h>

#include <villa_surface_detectors/PerceiveTableAction.h>
#include "../include/VillaSurfaceDetector.h"

#include <pcl_angle_filter.h>


#define TARGET_FRAME "map" //target frame name DONT CHANGE!
#define MAX_Z 2.0 // maximum z-value of point cloud in map frame to be considered
#define PLANE_CLUSTER_TOLERANCE 0.1
#define VISUALIZE true
#define XY_BUFFER 0.08
#define Z_BUFFER 0.08
#define DEFAULT_OBJECT_HEIGHT 0.30
#define TABLE_OBJECT_MARKER_NAMESPACE "table_object_marker"

bool compare_cluster_size(const pcl::PointIndices &lhs, const pcl::PointIndices &rhs) {
  return lhs.indices.size() < rhs.indices.size();
}

double calculate_density(sensor_msgs::PointCloud2 cloud, const visualization_msgs::Marker &box) {
	// TODO: Calculate true volume
	// If the cloud is one point thick in some dimension, we'll assign that dimension a magnitude of 1cm
	double volume = box.scale.x * box.scale.y * box.scale.z;
	return (double)(cloud.height * cloud.width) / volume;
}

class PerceiveTableAction
{
protected:

  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<villa_surface_detectors::PerceiveTableAction> as_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
  std::string action_name_;
  // create messages that are used to published feedback/result
  villa_surface_detectors::PerceiveTableFeedback feedback_;
  villa_surface_detectors::PerceiveTableResult result_;

  // VillaSurfaceDetector
  VillaSurfaceDetector vsd_;

  ros::Publisher detected_table_marker_pub;
  ros::Publisher table_points_pub;
  ros::Publisher object_points_pub;
  ros::Publisher object_marker_array_pub;

public:

  PerceiveTableAction(std::string name) :
    as_(nh_, name, boost::bind(&PerceiveTableAction::executeCB, this, _1), false),
    action_name_(name)
  {
    as_.start();
    ros::NodeHandle pnh("~");
    vsd_.init(pnh);

    if (VISUALIZE) {
      detected_table_marker_pub = pnh.advertise<visualization_msgs::Marker>("marker", 1, true);
      table_points_pub = pnh.advertise<sensor_msgs::PointCloud2>("points", 1, true);
      object_points_pub = pnh.advertise<sensor_msgs::PointCloud2>("objects/points", 1, true);
      object_marker_array_pub = pnh.advertise<visualization_msgs::MarkerArray>("objects/markers", 1, true);
    }
  }

  ~PerceiveTableAction(void)
  {
  }

  void executeCB(const villa_surface_detectors::PerceiveTableGoalConstPtr &goal)
  {
    bool success;

    ROS_INFO("Extracting table location from given point cloud...");
  	// Find Horizontal Planes from the given point cloud

    PointCloudT::Ptr converted_cloud (new PointCloudT);
    pcl::fromROSMsg(goal->cloud_input, *converted_cloud);
    converted_cloud->header.frame_id = goal->cloud_input.header.frame_id;

    float start_angle = goal->ignore_start_angle.data;
    float end_angle = goal->ignore_end_angle.data;
    PointCloudT::Ptr filtered_cloud (new PointCloudT);
    // Filter out a pie slice of the cloud, if the user asked for it
    if (start_angle != end_angle) {
      Eigen::Vector2f origin(goal->ignore_origin.x, goal->ignore_origin.y);
      filter_points_between(converted_cloud, filtered_cloud, origin, start_angle, end_angle);
      filtered_cloud->header = converted_cloud->header;
      converted_cloud = filtered_cloud;
    } else {
      pcl::fromROSMsg(goal->cloud_input, *filtered_cloud);
    }

    Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0);
    std::vector<PointCloudT> horizontal_planes_out;
    std::vector<geometry_msgs::Quaternion> horizontal_plane_coefs_out;
    std::vector<visualization_msgs::Marker> horizontal_plane_bounding_boxes_out;
    std::vector<visualization_msgs::Marker> horizontal_plane_AA_bounding_boxes_out;
    ROS_INFO("Calling detect_horizontal_planes function to extract horizontal planes");

    if (vsd_.detect_horizontal_planes(axis, *filtered_cloud, horizontal_planes_out,
    horizontal_plane_coefs_out, horizontal_plane_bounding_boxes_out,
    horizontal_plane_AA_bounding_boxes_out)) {
  		ROS_INFO("detect_horizontal_planes call was successful!");
  		if (!horizontal_planes_out.empty()){
  			ROS_INFO("Detected %zu planes", horizontal_planes_out.size());
  		} else {
  			ROS_INFO("No Planes Detected. Table cannot be extracted");
        as_.setAborted();
        return;
      }
  	} else {
  		ROS_ERROR("Failed to call service detect_horizontal_planes");
      as_.setAborted();
      return;
  	}

    // Sort by the surface area (XY) of the planes
    // Keep the indices around so we can get the other metadata we need
    vector< pair<int, float> > index_with_size;
    for (int i = 0; i < horizontal_planes_out.size(); i++) {
        const visualization_msgs::Marker &box = horizontal_plane_bounding_boxes_out[i];
        index_with_size.push_back(pair<int, float>(i, box.scale.x * box.scale.y));
        ROS_INFO("Area of %f", box.scale.x * box.scale.y);
    }

    const pair<int, float> largest = *std::max_element(index_with_size.begin(), index_with_size.end(), compare_by_second);
    const int largest_index = largest.first;
    PointCloudT::Ptr table_points = horizontal_planes_out[largest_index].makeShared();

    ROS_INFO("Largest Plane has index %i and has size of %f at height %f", largest_index, (float)largest.second, horizontal_plane_bounding_boxes_out[largest_index].pose.position.z);

    std::vector<PointCloudT::Ptr> objects_out;
    std::vector<visualization_msgs::Marker> object_bounding_boxes_out;
    if (vsd_.segment_objects(axis, filtered_cloud, horizontal_plane_coefs_out[largest_index], horizontal_plane_bounding_boxes_out[largest_index]\
      , DEFAULT_OBJECT_HEIGHT, objects_out, object_bounding_boxes_out)) {
      ROS_INFO("Detected %zu objects on this table", objects_out.size());
    } else {
      ROS_INFO("No objects found on this table");
    }

    // Display the objects' pointclouds
    if (VISUALIZE){
      ROS_INFO("Publishing objects pointclouds.");
      PointCloudT objects_viz_points;
      std::vector<PointCloudT::Ptr>::const_iterator it = objects_out.begin();
      if (it == objects_out.end()){
        ROS_INFO("No object pointclouds to publish.");
      } else {
        objects_viz_points += **it;
        objects_viz_points.header.frame_id = (*it)->header.frame_id;
        it++;
      }
      while (it < objects_out.end()){
        objects_viz_points += **it;
        it++;
      }
      visualization_msgs::MarkerArray object_marker_array;
      for (size_t i = 0; i < object_bounding_boxes_out.size(); i++) {
        visualization_msgs::Marker marker = object_bounding_boxes_out[i];
        ROS_INFO("Object bounding_box %zu added", i);

    		// Modify Marker
    		marker.ns = TABLE_OBJECT_MARKER_NAMESPACE;
    		marker.id = (int)i;
    		marker.color.r = 0; marker.color.g = 1.0; marker.color.b = 0;
    		object_marker_array.markers.push_back( marker ); // for visualization
      }
      // object_marker_array.markers = object_marker_array;
      table_points_pub.publish(horizontal_planes_out[largest_index]);
      detected_table_marker_pub.publish(horizontal_plane_bounding_boxes_out[largest_index]);
      object_points_pub.publish(objects_viz_points);
      object_marker_array_pub.publish(object_marker_array);
    }

    //Load data into result object to return
    sensor_msgs::PointCloud2 table_cloud;
    pcl::toROSMsg(horizontal_planes_out[largest_index], table_cloud);
    std::vector<sensor_msgs::PointCloud2> sensor_msgs_objects;
    for (std::vector<PointCloudT::Ptr>::const_iterator it = objects_out.begin(); it != objects_out.end(); it++) {
      sensor_msgs::PointCloud2 ros_object;
      pcl::toROSMsg(**it, ros_object);
      sensor_msgs_objects.push_back(ros_object);
    }

    result_.table_points = table_cloud;
    result_.table_coefs = horizontal_plane_coefs_out[largest_index];
    result_.table_bounding_box = horizontal_plane_bounding_boxes_out[largest_index];
    result_.objects = sensor_msgs_objects;
    result_.object_bounding_boxes = object_bounding_boxes_out;

    as_.setSucceeded(result_);
  }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "perceive_table");

  PerceiveTableAction percive_table("perceive_table");
  ROS_INFO("table detector action server online");
  ros::spin();

  return 0;
}
