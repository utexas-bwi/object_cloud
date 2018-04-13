#include "../include/VillaSurfaceDetector.h"

#include <signal.h>
#include <vector>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>

// PCL specific includes

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/console/parse.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>

// #include "villa_surface_detectors/DetectHorizontalPlanes.h"

#include "../include/BoundingBox.h"
#include <visualization_msgs/MarkerArray.h>


#define TARGET_FRAME "map" //target frame name DONT CHANGE!
#define EPS_ANGLE 0.05 //epsilon angle for segmenting, value in radians
#define VOXEL_LEAF_SIZE 0.02 //size of voxel leaf for processing
#define RANSAC_MAX_ITERATIONS 1000
#define PLANE_DIST_TRESH 0.025 //maximum distance from plane
#define CLUSTER_TOL 0.05 //clustering tolerance for largest plane extraction
#define MIN_NUMBER_PLANE_POINTS 500
#define MIN_PLANE_DENSITY 35000
#define STOPPING_PERCENTAGE 0.25 // Stop once we've processed all but X percentage of the cloud
#define IGNORE_FLOOR true // If the input cloud doesn't already have z filtered, we can do it
#define MIN_Z 0.05 // minimum z-value of point cloud in map frame to be considered
#define MAX_Z 2.0 // maximum z-value of point cloud in map frame to be considered

#define VISUALIZE false // NOT SUPPORTED RIGHT NOW
#define DEBUG_ENTER false // if true, you have to press enter to continue the process

using namespace std;

// publishers (not initialized!)
ros::Publisher voxel_cloud_pub;
ros::Publisher horizontal_plane_marker_pub;
ros::Publisher current_plane_cloud_pub;
ros::Publisher remaining_cloud_pub;

// bool compare_cluster_size(const pcl::PointIndices &lhs, const pcl::PointIndices &rhs) {
//   return lhs.indices.size() > rhs.indices.size();
// }

/*Function for finding the largest plane from the segmented "table"
 * removes noise*/
std::vector<PointCloudT::Ptr> VillaSurfaceDetector::isolate_surfaces(const PointCloudT::Ptr &in, double tolerance){
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (in);
    ROS_INFO("point cloud size of 'plane cloud' : %ld", in->size());

    //use euclidean cluster extraction to eliminate noise and get largest plane
    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (tolerance);
    ec.setMinClusterSize (MIN_NUMBER_PLANE_POINTS);
    ec.setSearchMethod (tree);
    ec.setInputCloud (in);
    ec.extract (cluster_indices);

    ROS_INFO("number of 'plane clouds' : %ld", cluster_indices.size());

    if (cluster_indices.empty()) {
        throw std::exception();
    }
		ROS_INFO("    Total number of surfaces from plane: %zu", cluster_indices.size());

    //pcl::PointIndices largest_cluster_indices = *std::max_element(cluster_indices.begin(), cluster_indices.end(), compare_cluster_size);
		std::vector<PointCloudT::Ptr> surfaces;

		//Optional
		//std::sort(cluster_indices, compare_cluster_size);
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it < cluster_indices.end(); it++) {
			PointCloudT::Ptr cluster(new PointCloudT(*in, (*it).indices));
	    surfaces.push_back(cluster);
		}
		return surfaces;
}

bool VillaSurfaceDetector::detect_horizontal_planes(Eigen::Vector3f axis, const PointCloudT &cloud_input,
  vector<PointCloudT> &horizontal_planes_out,
  vector<geometry_msgs::Quaternion> &horizontal_plane_coefs_out,
  vector<visualization_msgs::Marker> &horizontal_plane_bounding_boxes_out,
  vector<visualization_msgs::Marker> &horizontal_plane_AA_bounding_boxes_out){

	ROS_INFO("detect_horizontal_planes function started");
	if (cloud_input.points.empty()){
		ROS_ERROR("The input cloud is empty. Cannot process service");
		return false;
	}

	// Convert cloud to map frame:

	// ROS_INFO("detect_horizontal_planes: convert cloud to map frame");
	// // Define Frame Parameters
	// string z_axis_reference_frame (Z_AXIS_REFERENCE_FRAME);
	// string target_frame (TARGET_FRAME);
	//
  // PointCloudT::Ptr map_cloud(new PointCloudT);
	// PointCloudT::Ptr cloud_input_ptr = cloud_input.makeShared();
	// ROS_INFO("detect_horizontal_planes: move_to_frame");
  // this->move_to_frame((cloud_input_ptr), TARGET_FRAME, map_cloud);

	PointCloudT::Ptr map_cloud = cloud_input.makeShared();

	// If the floor is to be ignored, use the filtered map instead
	if (IGNORE_FLOOR){
		ROS_INFO("detect_horizontal_planes: filtering floor");
    // Create filtered point cloud
    PointCloudT::Ptr map_cloud_pass_through (new PointCloudT);

    // Create the filtering object
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud (map_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (MIN_Z, MAX_Z);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*map_cloud_pass_through);
    map_cloud = map_cloud_pass_through;
	}

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	ROS_INFO("detect_horizontal_planes: downsample input");
	PointCloudT::Ptr cloud_filtered (new PointCloudT);
	pcl::VoxelGrid<PointT> vg;
	vg.setInputCloud (map_cloud);
	// TODO: Do we need as much resolution in XY as Z?
	vg.setLeafSize (VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE);
	vg.filter (*cloud_filtered);

	if (VISUALIZE) {
		//debugging publishing
		sensor_msgs::PointCloud2 ros_cloud;
		pcl::toROSMsg(*cloud_filtered, ros_cloud);
		ros_cloud.header.frame_id = map_cloud->header.frame_id;
		ROS_INFO("Visualizing pointcloud to extract horizontal planes from...");
		voxel_cloud_pub.publish(ros_cloud);
	}
	if (DEBUG_ENTER){
    this->pressEnter("    Press ENTER to continue");
	}

	//create objects for use in segmenting
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	PointCloudT::Ptr cloud_plane (new PointCloudT);
	PointCloudT::Ptr cloud_remainder (new PointCloudT);

	// Create the segmentation object
	ROS_INFO("detect_horizontal_planes: segmentation");
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients (true);

	//look for a plane perpendicular to a given axis
	seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (RANSAC_MAX_ITERATIONS);
	seg.setDistanceThreshold (PLANE_DIST_TRESH);
	seg.setEpsAngle(EPS_ANGLE);

	//create the axis to use
	// ROS_INFO("Finding planes perpendicular to z-axis of %s frame",  z_axis_reference_frame.c_str());
	// geometry_msgs::Vector3Stamped ros_vec;
	// ros_vec.header.frame_id = z_axis_reference_frame;
	// ros_vec.vector.x = 0.0;
	// ros_vec.vector.y = 0.0;
	// ros_vec.vector.z = 1.0;
	//
  // geometry_msgs::Vector3Stamped out_vec;
  // this->vector_to_frame(ros_vec, map_cloud->header.frame_id, out_vec);
	//
	// //set the axis to the transformed vector
	// Eigen::Vector3f axis = Eigen::Vector3f(out_vec.vector.x, out_vec.vector.y , out_vec.vector.z);
	seg.setAxis(axis);
	ROS_INFO("SAC axis value: %f, %f, %f", seg.getAxis()[0], seg.getAxis()[1], seg.getAxis()[2]);

	// Create the filtering object
	pcl::ExtractIndices<PointT> extract;

	// Prepare object containers for responses
	vector<PointCloudT> horizontal_planes;
	vector<Eigen::Vector4f> horizontal_plane_coefs;
	vector<visualization_msgs::Marker> horizontal_plane_bounding_boxes_markers;
	vector<visualization_msgs::Marker> horizontal_plane_AA_bounding_boxes_markers;
	vector<pair<int, float> > indices_with_densities;
	int num_planes = 0;
	int ransac_iter = 0;
	size_t num_start_points = cloud_filtered->points.size();

	while (cloud_filtered->points.size() >  STOPPING_PERCENTAGE * num_start_points) {
		// Segment the largest planar component from the remaining cloud
		ROS_INFO("Extracting a horizontal plane %d", ransac_iter + 1);
		seg.setInputCloud (cloud_filtered);
		ROS_INFO("    Number of Points to Process: %zu", cloud_filtered->size());

		seg.segment (*inliers, *coefficients);
		if (inliers->indices.empty()){
		  ROS_WARN("    Could not estimate a planar model for the given dataset.");
		  break;
		}

		ROS_INFO("    Found a horizontal plane!");
		// Extract the inliers
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);
		extract.filter (*cloud_plane);


		// Create the filtering object to extract everything else
		extract.setNegative (true);
		extract.filter (*cloud_remainder);
		cloud_filtered.swap (cloud_remainder);

		if (VISUALIZE) {
			//debugging publishing
			sensor_msgs::PointCloud2 ros_remainder;
			pcl::toROSMsg(*cloud_remainder, ros_remainder);
			ros_remainder.header.frame_id = map_cloud->header.frame_id;
			remaining_cloud_pub.publish(ros_remainder);

			sensor_msgs::PointCloud2 ros_cloud;
			pcl::toROSMsg(*cloud_plane, ros_cloud);
			ros_cloud.header.frame_id = map_cloud->header.frame_id;
			current_plane_cloud_pub.publish(ros_cloud);

			ROS_INFO("Entire Plane Published");
			if (DEBUG_ENTER) {
				this->pressEnter("		Press Enter to begin showing surfaces");
			}
		}

		// Perform clustering on this plane.
		std::vector<PointCloudT::Ptr> surfaces;

		// if no clusters are found, this is an invalid plane extraction
		try{
			double cluster_extraction_tolerance = CLUSTER_TOL;
			surfaces = this->isolate_surfaces(cloud_plane, cluster_extraction_tolerance);

			// if (cloud_plane->size() < MIN_NUMBER_PLANE_POINTS){
			// 	ROS_WARN("Plane contains insufficient points. Discarding");
			// 	continue;
			// }

		}
		catch(std::exception &e){
			ROS_WARN("No clusters were found. Invalid plane points exist in this iteration.");
			continue;
		}

		// if (VISUALIZE) {
		// 	sensor_msgs::PointCloud2 horizontal_plane_cloud_ros;
		// 	pcl::toROSMsg(*cloud_plane, horizontal_plane_cloud_ros);
		// 	horizontal_plane_cloud_ros.header.frame_id = map_cloud->header.frame_id;
		// 	// Now that we've pulled out the largest cluster, lets visualize
		// 	current_plane_cloud_pub.publish(horizontal_plane_cloud_ros);
		// }

		for (std::vector<PointCloudT::Ptr>::const_iterator it = surfaces.begin(); it < surfaces.end(); it++) {
			PointCloudT::Ptr current = *it;

			//get the plane coefficients
			Eigen::Vector4f plane_coefficients;
			plane_coefficients(0) = coefficients->values[0];
			plane_coefficients(1) = coefficients->values[1];
			plane_coefficients(2) = coefficients->values[2];
			plane_coefficients(3) = coefficients->values[3];

			// Extract the bonding box parameters of this plane
	    const BoundingBox &oriented_bbox_params = this->extract_oriented_bounding_box_params(current);

			// Use the oriented bounding box for a better estimate of density. Non oriented box
			// penalizes shelves that don't happen to be perfectly aligned with the map frame
		 	double plane_bounding_box_density = this->calculate_density(current,oriented_bbox_params);


		 	// Create Marker to represent bounding box
		 	visualization_msgs::Marker plane_bounding_box_marker;
		 	plane_bounding_box_marker = this->createBoundingBoxMarker(oriented_bbox_params, 0, TARGET_FRAME);

		 	//store each "horizontal_plane" found
			sensor_msgs::PointCloud2 horizontal_plane_cloud_ros;
			pcl::toROSMsg(*current, horizontal_plane_cloud_ros);
			horizontal_plane_cloud_ros.header.frame_id = map_cloud->header.frame_id;

			const BoundingBox &aa_bbbox_params = this->extract_aa_bounding_box_params(current);
		 	visualization_msgs::Marker plane_AA_bounding_box_marker = this->createBoundingBoxMarker(aa_bbbox_params, num_planes, TARGET_FRAME);

			if (VISUALIZE) {
				// Visualize plane bounding box marker
				current_plane_cloud_pub.publish(horizontal_plane_cloud_ros);
				horizontal_plane_marker_pub.publish(plane_bounding_box_marker);
			}

			if (plane_bounding_box_density < MIN_PLANE_DENSITY){
				ROS_INFO("Rejecting candidate plane with low density (%f)", plane_bounding_box_density);
				if (DEBUG_ENTER){
					ROS_INFO("%d planes total", num_planes);
					this->pressEnter("    Press ENTER to show next surface in plane");
				}
				continue;
			};

			horizontal_plane_coefs.push_back(plane_coefficients);
			horizontal_plane_bounding_boxes_markers.push_back(plane_bounding_box_marker);
			horizontal_planes.push_back(*current);
			horizontal_plane_AA_bounding_boxes_markers.push_back(plane_AA_bounding_box_marker);
			indices_with_densities.push_back(pair<int, float>(num_planes, (float) plane_bounding_box_density));
			num_planes += 1;

			if (DEBUG_ENTER){
				ROS_INFO("%d planes total", num_planes);
				this->pressEnter("    Press ENTER to show next surface in plane");
			}
		}

		ransac_iter++;
		if (DEBUG_ENTER) {
			this->pressEnter("		Press Enter to show next entire plane");
		}
	}

  std::sort(indices_with_densities.begin(), indices_with_densities.end(), compare_by_second);

	// Populate the response with planes sorted by density
	for (vector< pair<int, float> >::const_iterator it = indices_with_densities.begin(); it < indices_with_densities.end(); ++it){
		pair<int, float> pair = *it;
		int i = pair.first;
		float density = pair.second;

		horizontal_planes_out.push_back(horizontal_planes.at(i));
		geometry_msgs::Quaternion coef;
		Eigen::Vector4f current = horizontal_plane_coefs[i];
		coef.x = current(0);
		coef.y = current(1);
		coef.z = current(2);
		coef.w = current(3);

		ROS_INFO("Plane %i ori: x:%f, y:%f, z:%f, density: %f", i + 1, horizontal_plane_bounding_boxes_markers.at(i).pose.position.x, horizontal_plane_bounding_boxes_markers.at(i).pose.position.y, horizontal_plane_bounding_boxes_markers.at(i).pose.position.z, density);

		horizontal_plane_coefs_out.push_back(coef);
		horizontal_plane_bounding_boxes_out.push_back(horizontal_plane_bounding_boxes_markers.at(i));
		horizontal_plane_AA_bounding_boxes_out.push_back(horizontal_plane_AA_bounding_boxes_markers.at(i));
	}

	ROS_INFO("Horizontal Plane Segmentation finished: %zu planes of at least %d density found", horizontal_planes_out.size(), MIN_PLANE_DENSITY);

	return true;
}
