#include "../include/VillaSurfaceDetector.h"

#define BOUNDING_BOX_MARKER_NAMESPACE "horizontal_planes_marker"

using namespace std;

void VillaSurfaceDetector::pressEnter(string message){
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

visualization_msgs::Marker VillaSurfaceDetector::createBoundingBoxMarker(const BoundingBox &box,
												   const int &marker_index, string frame){
    visualization_msgs::Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = BOUNDING_BOX_MARKER_NAMESPACE;
    marker.id = marker_index;

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::CUBE;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header

    marker.pose.position.x = box.position[0];
    marker.pose.position.y = box.position[1];
    marker.pose.position.z = box.position[2];
	marker.pose.orientation.x = (double)box.orientation.x();
	marker.pose.orientation.y = (double)box.orientation.y();
	marker.pose.orientation.z = (double)box.orientation.z();
	marker.pose.orientation.w = (double)box.orientation.w();

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = abs(box.max[0] - box.min[0]);
    marker.scale.y = abs(box.max[1] - box.min[1]);
    marker.scale.z = abs(box.max[2] - box.min[2]);

		ROS_INFO("Marker made with dimensions x = %f, y = %f, z = %f", marker.scale.x, marker.scale.y, marker.scale.z);

    // Set the color -- be sure to set alpha to something non-zero!
    //int	apply_color = marker_index % colormap.size();

    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 0.25;

    marker.lifetime = ros::Duration();

    return marker;
}

BoundingBox VillaSurfaceDetector::extract_aa_bounding_box_params(const PointCloudT::Ptr &plane_cloud){
  Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
  pcl::compute3DCentroid(*plane_cloud, centroid);
  Eigen::Vector4f min, max = Eigen::Vector4f::Zero();
  pcl::getMinMax3D(*plane_cloud, min, max);

	Eigen::Vector4f position((max[0] - min[0]) / 2, (max[1] - min[2]) / 2, (max[3] - min[3]) / 2, 1);
	return BoundingBox(min, max, centroid, Eigen::Quaternionf(0,0,0,1), position, 0);
}

BoundingBox VillaSurfaceDetector::extract_oriented_bounding_box_params(const PointCloudT::Ptr &plane_cloud){
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*plane_cloud, centroid);

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud (plane_cloud);
	feature_extractor.compute();

	PointT min, max, position;
	Eigen::Matrix3f rotational_matrix;
	feature_extractor.getOBB(min, max, position, rotational_matrix);
	Eigen::Quaternionf quat(rotational_matrix);
	// Homogeneous coordinates
	Eigen::Vector4f min_vec(min.x, min.y, min.z, 1);
	Eigen::Vector4f max_vec(max.x, max.y, max.z, 1);
	Eigen::Vector4f position_vec(position.x, position.y, position.z, 1);

  return BoundingBox(min_vec, max_vec, centroid, quat, position_vec, 0);
}

void VillaSurfaceDetector::move_to_frame(const PointCloudT::Ptr &input, const string &target_frame,
	PointCloudT::Ptr &output) {
    ROS_INFO("Transforming Input Point Cloud to %s frame...",  target_frame.c_str() );
    ROS_INFO("    Input Cloud Size: %zu", input->size());
    if (input->header.frame_id == target_frame) {
        output = input;
        return;
    }
    while (ros::ok()){
        tf::StampedTransform stamped_transform;
        try{
            // Look up transform
            tf_listener_->lookupTransform(target_frame, input->header.frame_id, ros::Time(0), stamped_transform);
						ROS_INFO("Transfoming from (%s) to (%s)", input->header.frame_id.c_str(), target_frame.c_str());
            // Apply transform
            pcl_ros::transformPointCloud(*input, *output, stamped_transform);

            // Store Header Details
            output->header.frame_id = target_frame;
            pcl_conversions::toPCL(ros::Time::now(), output->header.stamp);

            break;
        }
            //keep trying until we get the transform
        catch (tf::TransformException &ex){
            ROS_ERROR_THROTTLE(1, "%s", ex.what());
            ROS_WARN_THROTTLE(1,"    Waiting for transform from cloud frame (%s) to %s frame. Trying again", input->header.frame_id.c_str(), target_frame.c_str());
            continue;
        }
    }
}

void VillaSurfaceDetector::vector_to_frame(const geometry_msgs::Vector3Stamped &vector,
	const string &target_frame,geometry_msgs::Vector3Stamped &out_vector) {
	if (vector.header.frame_id == target_frame) {
  	out_vector = vector;
    return;
  }
	string z_axis_reference_frame (Z_AXIS_REFERENCE_FRAME);
  //transform into the camera's frame
  while (ros::ok()){
  	tf::StampedTransform transform;
    try{
    	tf_listener_->transformVector(target_frame, ros::Time(0), vector, z_axis_reference_frame, out_vector);
      break;
    }
    //keep trying until we get the transform
    catch (tf::TransformException ex){
      ROS_ERROR_THROTTLE(1, "%s", ex.what());
      ROS_WARN_THROTTLE(1, "   Waiting for tf to transform desired SAC axis to point cloud frame. trying again");
      continue;
    }
  }
}

double VillaSurfaceDetector::calculate_density(const PointCloudT::Ptr &cloud, const BoundingBox &box) {
	// TODO: Calculate true volume
	// If the cloud is one point thick in some dimension, we'll assign that dimension a magnitude of 1cm
	double x_dim = max(abs(box.max[0] - box.min[0]), 0.01f);
	double y_dim = max(abs(box.max[1] - box.min[1]), 0.01f);
	double z_dim = max(abs(box.max[2] - box.min[2]), 0.01f);
	double volume = x_dim * y_dim * z_dim;
	return (double)cloud->size() / volume;
}
