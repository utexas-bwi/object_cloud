#include <algorithm>

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>

// PCL specific includes
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <villa_surface_detectors/PerceiveShelfAction.h>
#include "../include/VillaSurfaceDetector.h"

#include <pcl_angle_filter.h>


#define TARGET_FRAME "map" //target frame name DONT CHANGE!
#define CUPBOARD_MARKER_NAMESPACE "cupboard_marker"
#define CUPBOARD_SHELVES_MARKER_NAMESPACE "cupboard_shelf_marker"
#define SHELF_OBJECT_MARKER_NAMESPACE "shelf_object_marker"
#define MAX_Z 2.0 // maximum z-value of point cloud in map frame to be considered
#define MIN_NUM_CUPBOARD_PLANES 2
#define PLANE_CLUSTER_TOLERANCE 0.1
#define VISUALIZE true
#define XY_BUFFER 0.08
#define Z_BUFFER 0.08
#define RANSAC_MAX_ITERATIONS 1000
#define DEFAULT_OBJECT_HEIGHT 0.30
#define VERT_PLANE_IGNORE_HEIGHT 0.5
#define VERT_PLANE_DIST_TRESH 0.02
#define STOPPING_PERCENTAGE 0.10
#define VERT_PLANE_CLUSTER_TOLERANCE 0.1

typedef struct {
  sensor_msgs::PointCloud2 cloud;
  visualization_msgs::Marker oriented_bounding_box;
  geometry_msgs::Quaternion horizontal_plane_coefs;
} ShelfData;

void pressEnter(string message){
	std::cout << message;
	while (true){
		char c = std::cin.get();
		if (c == '\n')
			break;
		else if (c == 'q'){
			ros::shutdown();
			exit(1);
		}
		else {
			std::cout <<  message;
		}
	}
}

bool compare_shelf_height(const ShelfData &lhs, const ShelfData &rhs) {
  return lhs.oriented_bounding_box.pose.position.z < rhs.oriented_bounding_box.pose.position.z;
}

bool compare_cluster_size(const pcl::PointIndices &lhs, const pcl::PointIndices &rhs) {
  return lhs.indices.size() < rhs.indices.size();
}


bool compare_marker_z (const visualization_msgs::Marker &lhs, const visualization_msgs::Marker &rhs) {
	double lhs_dist = lhs.pose.position.z;
	double rhs_dist = rhs.pose.position.z;
	return lhs_dist < rhs_dist;
}

double calculate_density(sensor_msgs::PointCloud2 cloud, const visualization_msgs::Marker &box) {
	// TODO: Calculate true volume
	// If the cloud is one point thick in some dimension, we'll assign that dimension a magnitude of 1cm
	double volume = box.scale.x * box.scale.y * box.scale.z;
	return (double)(cloud.height * cloud.width) / volume;
}

/*Function for finding the largest collection of plane centers projected in the xy plane*/
pcl::PointIndices get_largest_cluster(const PointCloudT::Ptr &in, double tolerance){
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (in);
    ROS_INFO("point cloud size of 'plane cloud' : %ld", in->size());

    //use euclidean cluster extraction to eliminate noise and get largest plane
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (tolerance); // Points within this radius of eachother will be aggregated as a single point
    ec.setMinClusterSize (MIN_NUM_CUPBOARD_PLANES);
    ec.setSearchMethod (tree);
    ec.setInputCloud (in);
    ec.extract (cluster_indices);

    ROS_INFO("number of 'plane clouds' : %ld", cluster_indices.size());

    if (cluster_indices.empty()) {
        throw std::exception();
    }

    return *std::max_element(cluster_indices.begin(), cluster_indices.end(), compare_cluster_size);
}

PointCloudT::Ptr filter_vertical_planes(const PointCloudT::Ptr cloud_in, ros::Publisher shelves_points_pub) {
  //create objects for use in segmenting
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  PointCloudT::Ptr cloud_filtered (new PointCloudT);
  PointCloudT::Ptr cloud_plane (new PointCloudT);
  PointCloudT::Ptr cloud_remainder (new PointCloudT);
  PointCloudT::Ptr cloud_to_remove (new PointCloudT);
  PointCloudT::Ptr cloud_out (new PointCloudT);
  cloud_to_remove->header = cloud_in->header;
  *cloud_filtered = *cloud_in;
  *cloud_out = *cloud_in;

  // Create the segmentation object
  ROS_INFO("filtering vertical planes larger than %f", VERT_PLANE_IGNORE_HEIGHT);
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients (true);

  //look for dominant planes
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (RANSAC_MAX_ITERATIONS);
  seg.setDistanceThreshold (VERT_PLANE_DIST_TRESH);

  // Create the filtering object
  pcl::ExtractIndices<PointT> extract;

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(VERT_PLANE_CLUSTER_TOLERANCE);
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod(tree);
  // pcl::PointIndices::Ptr all_vertical_surfaces(new pcl::PointIndices());

  int ransac_iter = 0;
	size_t num_start_points = cloud_filtered->points.size();
  while(cloud_filtered->points.size() > STOPPING_PERCENTAGE * num_start_points) {
    ROS_INFO("Extracting a plane %d", ransac_iter + 1);
		seg.setInputCloud (cloud_filtered);
		ROS_INFO("    Number of Points to Process: %zu", cloud_filtered->size());

    seg.segment (*inliers, *coefficients);
		if (inliers->indices.empty()){
		  ROS_WARN("    Could not estimate a planar model for the given dataset.");
		  break;
		}

    // Extract the inliers
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);
		extract.filter (*cloud_plane);

		// Create the filtering object to extract everything else
		extract.setNegative (true);
		extract.filter (*cloud_remainder);
		*cloud_filtered = *cloud_remainder;

    std::cout << "Model coefficients: " << coefficients->values[0] << " "
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " "
                                      << coefficients->values[3] << std::endl;
    bool vertical = std::abs(coefficients->values[2]) < 0.03;

    if (VISUALIZE) {
      ROS_INFO("Publishing plane, vertical = %d", vertical);
      sensor_msgs::PointCloud2 cloud_plane_ros;
      pcl::toROSMsg(*cloud_plane, cloud_plane_ros);
      cloud_plane_ros.header.frame_id = cloud_plane->header.frame_id;
      shelves_points_pub.publish(cloud_plane_ros);
      pressEnter("Press enter to segment next plane");
    }

    if (vertical) {
      cluster_indices.clear();
      tree->setInputCloud(cloud_plane);
      ec.setInputCloud(cloud_plane);
      ec.extract(cluster_indices);

      // ROS_INFO("Size of kdtree %zu", tree->getIndices()->size());
      ROS_INFO("Number of clusters from plane: %zu", cluster_indices.size());
      for (int i = 0; i < cluster_indices.size(); i++) {
        PointCloudT::Ptr cluster(new PointCloudT(*cloud_plane, cluster_indices[i].indices));
        PointT min = PointT();
        PointT max = PointT();
        pcl::getMinMax3D(*cluster, min, max);
        float height = max.z - min.z;
        float width = std::max(max.x - min.x, max.y - min.y);
        ROS_INFO("Height of cluster %d: %f", i + 1, height);
        ROS_INFO("Density of cluster %d: %f", i + 1, cluster->size() / (height * width * 0.01)); // 1 cm for the depth
        if (VISUALIZE) {
          ROS_INFO("Publishing cluster");
          sensor_msgs::PointCloud2 cloud_plane_ros;
          pcl::toROSMsg(*cluster, cloud_plane_ros);
          cloud_plane_ros.header.frame_id = cluster->header.frame_id;
          shelves_points_pub.publish(cloud_plane_ros);
          pressEnter("Press enter to segment next cluster");
        }
        if (max.z - min.z > VERT_PLANE_IGNORE_HEIGHT) {
          *cloud_to_remove += *cluster;
        }
      }
    }
    ransac_iter++;
  }
  ROS_INFO("Beginning to remove %zu total points", cloud_to_remove->size());

  pcl::KdTreeFLANN<PointT> flann_tree;
  flann_tree.setInputCloud(cloud_out);

  float nan = std::numeric_limits<float>::quiet_NaN();

  if (VISUALIZE) {
    ROS_INFO("Publishing cloud to remove");
    sensor_msgs::PointCloud2 cloud_plane_ros;
    pcl::toROSMsg(*cloud_to_remove, cloud_plane_ros);
    cloud_plane_ros.header.frame_id = cloud_in->header.frame_id;
    shelves_points_pub.publish(cloud_plane_ros);
    pressEnter("Press enter to remove from source cloud");
  }

  for (int i = 0; i < cloud_to_remove->size(); i++) {
    PointT searchPoint = cloud_to_remove->at(i);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    if (flann_tree.radiusSearch(searchPoint, 0.001, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
      // ROS_INFO("Search point: %f, %f, %f", searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++) {
        // ROS_INFO("Neighbor point: %f, %f, %f", cloud_out->at(pointIdxRadiusSearch[j]).x,  cloud_out->at(pointIdxRadiusSearch[j]).y,  cloud_out->at(pointIdxRadiusSearch[j]).z);
        cloud_out->at(pointIdxRadiusSearch[j]).x = nan;
        cloud_out->at(pointIdxRadiusSearch[j]).y = nan;
        cloud_out->at(pointIdxRadiusSearch[j]).z = nan;
        // cloud_out->cloud[pointIdxRadiusSearch[j]].r = nan;
        // cloud_out->cloud[pointIdxRadiusSearch[j]].g = nan;
        // cloud_out->cloud[pointIdxRadiusSearch[j]].b = nan;
      }
    }
  }

  // if (VISUALIZE) {
  //   ROS_INFO("Publishing pre nanremoved cloud");
  //   sensor_msgs::PointCloud2 cloud_plane_ros;
  //   pcl::toROSMsg(*cloud_out, cloud_plane_ros);
  //   cloud_plane_ros.header.frame_id = cloud_in->header.frame_id;
  //   shelves_points_pub.publish(cloud_plane_ros);
  //   pressEnter("Press enter to remove nan points");
  // }

  cloud_out->is_dense = false;
  std::vector<int> index;
  pcl::removeNaNFromPointCloud(*cloud_out, *cloud_out, index);

  ROS_INFO("From %zu points to %zu points", cloud_in->size(), cloud_out->size());

  return cloud_out;
}

PointCloudT::Ptr project_centers(const std::vector<visualization_msgs::Marker> &bounding_boxes){
    // Extract the centers of all the planes detected
    PointCloudT::Ptr cloud_plane_centers (new PointCloudT);
    unsigned long int i = 0;
    for(std::vector<visualization_msgs::Marker>::const_iterator it = bounding_boxes.begin(); it < bounding_boxes.end(); ++it){
        visualization_msgs::Marker bounding_box = *it;
        PointT box_center_pt;
        // Project them to the x-y plane only
        box_center_pt.x = bounding_box.pose.position.x;
        box_center_pt.y = bounding_box.pose.position.y;
        ROS_INFO("Plane %zu, Bounding Box Center Location: (%f, %f, %f)", i, box_center_pt.x, box_center_pt.y, box_center_pt.z );
        cloud_plane_centers->points.push_back(box_center_pt);
        i++;
    }
    return cloud_plane_centers;
}


visualization_msgs::Marker isolate_shelf(const std::vector<PointCloudT> horizontal_planes_out,
  const std::vector<geometry_msgs::Quaternion> horizontal_plane_coefs_out,
  const std::vector<visualization_msgs::Marker> horizontal_plane_bounding_boxes_out,
  const std::vector<visualization_msgs::Marker> horizontal_plane_AA_bounding_boxes_out,
  std::vector<ShelfData> &shelf_data_vector) {
    // Extract the centers of all the planes detected
    PointCloudT::Ptr cloud_plane_centers = project_centers(horizontal_plane_bounding_boxes_out);
    ROS_INFO("Cloud Plane Centers has size: %zu", cloud_plane_centers->size());

    pcl::PointIndices largest_cluster = get_largest_cluster(cloud_plane_centers, PLANE_CLUSTER_TOLERANCE);

    ROS_INFO("The largest cluster has %zu planes", largest_cluster.indices.size());

    int most_dense_index = 0;
    double highest_density = 0;

    size_t cluster_plane_size = largest_cluster.indices.size();
    for(size_t i = 0; i < cluster_plane_size; i++){
      int plane_index = largest_cluster.indices[i];

      visualization_msgs::Marker bbox = horizontal_plane_bounding_boxes_out[plane_index];
      sensor_msgs::PointCloud2 shelf_pointcloud_ros;
      pcl::toROSMsg(horizontal_planes_out[plane_index], shelf_pointcloud_ros);
      ShelfData shelf = {shelf_pointcloud_ros, bbox, horizontal_plane_coefs_out[plane_index]};
      shelf_data_vector.push_back(shelf);
      ROS_INFO("Bounding box with rotation: x = %f, y = %f, z = %f, w = %f and size x = %f, y = %f, z = %f"\
      , bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z, bbox.pose.orientation.w\
      , bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z);

      double cloud_density = calculate_density(shelf_data_vector[i].cloud, bbox);
      if (cloud_density > highest_density) {
        most_dense_index = i;
        highest_density = cloud_density;
      }
    }

    //Takes highest density plane that was percieved and uses it as the basis for the entire
    //shelf's orientation, position, and scale
    //TO DO: replace with something more robust like averaging angles etc.
    visualization_msgs::Marker shelf_marker = shelf_data_vector[most_dense_index].oriented_bounding_box;
    shelf_marker.scale.z = MAX_Z;
    shelf_marker.pose.position.z = MAX_Z / 2;
    shelf_marker.scale.x += XY_BUFFER;
    shelf_marker.scale.y += XY_BUFFER;

    return shelf_marker;
}


class PerceiveShelfAction
{
protected:

  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<villa_surface_detectors::PerceiveShelfAction> as_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
  std::string action_name_;
  // create messages that are used to published feedback/result
  villa_surface_detectors::PerceiveShelfFeedback feedback_;
  villa_surface_detectors::PerceiveShelfResult result_;

  // VillaSurfaceDetector
  VillaSurfaceDetector vsd_;

  ros::Publisher detected_cupboard_marker_pub;
  ros::Publisher cupboard_points_pub;
  ros::Publisher shelves_marker_array_pub;
  ros::Publisher shelves_points_pub;
  ros::Publisher object_points_pub;
  ros::Publisher object_marker_array_pub;

public:

  PerceiveShelfAction(std::string name) :
    as_(nh_, name, boost::bind(&PerceiveShelfAction::executeCB, this, _1), false),
    action_name_(name)
  {
    as_.start();
    ros::NodeHandle pnh("~");
    vsd_.init(pnh);

    if (VISUALIZE) {
      detected_cupboard_marker_pub = pnh.advertise<visualization_msgs::Marker>("marker", 1, true);
      cupboard_points_pub = pnh.advertise<sensor_msgs::PointCloud2>("points", 1, true);
      shelves_marker_array_pub = pnh.advertise<visualization_msgs::MarkerArray>("shelves/markers", 1, true);
      shelves_points_pub = pnh.advertise<sensor_msgs::PointCloud2>("shelves/points", 1, true);
      object_points_pub = pnh.advertise<sensor_msgs::PointCloud2>("objects/points", 1, true);
      object_marker_array_pub = pnh.advertise<visualization_msgs::MarkerArray>("objects/markers", 1, true);
    }
  }

  ~PerceiveShelfAction(void)
  {
  }

  void executeCB(const villa_surface_detectors::PerceiveShelfGoalConstPtr &goal)
  {
    ROS_INFO("Extracting shelf location from given point cloud...");
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

    // filtered_cloud = filter_vertical_planes(filtered_cloud, shelves_points_pub);
    // if (VISUALIZE) {
    //   ROS_INFO("Publishing filtered points to sheles/points temporarily");
    //   sensor_msgs::PointCloud2 filtered_cloud_ros;
    //   pcl::toROSMsg(*filtered_cloud, filtered_cloud_ros);
    //   filtered_cloud_ros.header.frame_id = filtered_cloud->header.frame_id;
    //   shelves_points_pub.publish(filtered_cloud_ros);
    //   pressEnter("Press enter to begin horizontal plane segmentation");
    // }

    Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0);
    std::vector<PointCloudT> horizontal_planes_out;
    std::vector<geometry_msgs::Quaternion> horizontal_plane_coefs_out;
    std::vector<visualization_msgs::Marker> horizontal_plane_bounding_boxes_out;
    std::vector<visualization_msgs::Marker> horizontal_plane_AA_bounding_boxes_out;
    ROS_INFO("Calling detect_horizontal_planes function to extract horizontal planes");

    if (vsd_.detect_horizontal_planes(axis,*filtered_cloud, horizontal_planes_out,
    horizontal_plane_coefs_out, horizontal_plane_bounding_boxes_out,
    horizontal_plane_AA_bounding_boxes_out)) {
  		ROS_INFO("detect_horizontal_planes call was successful!");
  		if (!horizontal_planes_out.empty()){
  			ROS_INFO("Detected %zu planes", horizontal_planes_out.size());
  		} else {
  			ROS_INFO("No Planes Detected. Cupboard cannot be extracted");
        as_.setAborted();
        return;
      }
  	} else {
  		ROS_ERROR("Failed to call service detect_horizontal_planes");
      as_.setAborted();
      return;
  	}

    std::vector<ShelfData> shelf_data_vector;

    visualization_msgs::Marker shelf_marker;
    try {
      shelf_marker = isolate_shelf(horizontal_planes_out,
        horizontal_plane_coefs_out, horizontal_plane_bounding_boxes_out,
        horizontal_plane_AA_bounding_boxes_out, shelf_data_vector);
    } catch (std::exception e) {
      ROS_INFO("No shelves found from the detected horizontal surfaces");
      as_.setAborted();
      return;
    }

    if (VISUALIZE) {
      ROS_INFO("Publishing marker bounding box for location of cupboard");
      detected_cupboard_marker_pub.publish(shelf_marker);
      PointCloudT::Ptr shelves_viz_points(new PointCloudT);

      visualization_msgs::MarkerArray shelf_marker_array;
      for (size_t i = 0; i < shelf_data_vector.size(); i++ ){
    		visualization_msgs::Marker shelf_marker = shelf_data_vector[i].oriented_bounding_box;

    		// Modify Marker
    		shelf_marker.ns = CUPBOARD_SHELVES_MARKER_NAMESPACE;
    		shelf_marker.id = (int)i;
    		shelf_marker.color.r = 0; shelf_marker.color.g = 1.0; shelf_marker.color.b = 0;
    		shelf_marker_array.markers.push_back( shelf_marker ); // for visualization
        shelf_data_vector[i].oriented_bounding_box = shelf_marker;

        // Accumulate all shelf point clouds into one cloud
        PointCloudT::Ptr shelf_points(new PointCloudT);
        pcl::fromROSMsg(shelf_data_vector[i].cloud, *shelf_points);
        *shelves_viz_points += *shelf_points;
    	}
      // Display visualization data
      shelves_marker_array_pub.publish(shelf_marker_array);
      sensor_msgs::PointCloud2 shelves_viz_points_ros;
      pcl::toROSMsg(*shelves_viz_points, shelves_viz_points_ros);
      shelves_viz_points_ros.header = shelf_data_vector[0].cloud.header;
      shelves_points_pub.publish(shelves_viz_points_ros);
      ROS_INFO("Publishing shelf markers and points");
    }

    //Sort shelf data lowest to heighest in preperation for object segmentation
    sort(shelf_data_vector.begin(), shelf_data_vector.end(), compare_shelf_height);

    std::vector<PointCloudT::Ptr> all_objects;
    std::vector<visualization_msgs::Marker> all_object_bounding_boxes;
    for (std::vector<ShelfData>::const_iterator it = shelf_data_vector.begin(); it < shelf_data_vector.end(); it++) {
      ROS_INFO("Shelf height: %f ", it->oriented_bounding_box.pose.position.z);
      std::vector<PointCloudT::Ptr> objects_out;
      std::vector<visualization_msgs::Marker> object_bounding_boxes_out;

      float height_difference = DEFAULT_OBJECT_HEIGHT;
      if (it + 1 != shelf_data_vector.end()) {
        height_difference = (*(it + 1)).oriented_bounding_box.pose.position.z - (*it).oriented_bounding_box.pose.position.z - Z_BUFFER;
      }
      ROS_INFO("Height difference is %f ", height_difference);

      if (vsd_.segment_objects(axis, filtered_cloud, it->horizontal_plane_coefs, it->oriented_bounding_box, height_difference, objects_out, object_bounding_boxes_out)) {
        ROS_INFO("Detected %zu objects on this shelf", objects_out.size());
      } else {
        ROS_INFO("No objects found on this shelf");
      }

      all_objects.insert(all_objects.end(), objects_out.begin(), objects_out.end());
      all_object_bounding_boxes.insert(all_object_bounding_boxes.end(), object_bounding_boxes_out.begin(), object_bounding_boxes_out.end());
    }

    // Display the objects' pointclouds
    if (VISUALIZE){
      ROS_INFO("Publishing objects pointclouds.");
      PointCloudT objects_viz_points;
      std::vector<PointCloudT::Ptr>::const_iterator it = all_objects.begin();
      if (it == all_objects.end()){
        ROS_INFO("No object pointclouds to publish.");
      } else {
        objects_viz_points += **it;
        objects_viz_points.header.frame_id = (*it)->header.frame_id;
        it++;
      }
      while (it < all_objects.end()){
        objects_viz_points += **it;
        it++;
      }
      visualization_msgs::MarkerArray object_marker_array;
      for (size_t i = 0; i < all_object_bounding_boxes.size(); i++) {
        visualization_msgs::Marker marker = all_object_bounding_boxes[i];
        ROS_INFO("Object bounding_box %zu added", i);

    		// Modify Marker
    		marker.ns = SHELF_OBJECT_MARKER_NAMESPACE;
    		marker.id = (int)i;
    		marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0;
    		object_marker_array.markers.push_back( marker ); // for visualization
      }
      object_points_pub.publish(objects_viz_points);
      object_marker_array_pub.publish(object_marker_array);
    }

    std::vector<sensor_msgs::PointCloud2> sensor_msgs_objects;
    for (std::vector<PointCloudT::Ptr>::const_iterator it = all_objects.begin(); it != all_objects.end(); it++) {
      sensor_msgs::PointCloud2 ros_object;
      pcl::toROSMsg(**it, ros_object);
      sensor_msgs_objects.push_back(ros_object);
    }

    //Load data into result object to return
    for (std::vector<ShelfData>::const_iterator it = shelf_data_vector.begin(); it < shelf_data_vector.end(); it++) {
      result_.shelves.push_back((*it).cloud);
      result_.shelf_coefs.push_back((*it).horizontal_plane_coefs);
      result_.shelf_bounding_boxes.push_back((*it).oriented_bounding_box);
    }
    result_.shelf_bounding_box = shelf_marker;
    result_.objects = sensor_msgs_objects;
    result_.object_bounding_boxes = all_object_bounding_boxes;

    as_.setSucceeded(result_);
  }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "perceive_shelf");

  PerceiveShelfAction percive_shelf("perceive_shelf");
  ROS_INFO("shelf detector action server online");
  ros::spin();

  return 0;
}
