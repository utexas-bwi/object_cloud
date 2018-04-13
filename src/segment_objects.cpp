#include <ros/ros.h>
#include <Eigen/Dense>
#include <visualization_msgs/Marker.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include "../include/VillaSurfaceDetector.h"

#define TARGET_FRAME "map" //target frame name DONT CHANGE!
#define BOUNDING_BOX_MARKER_NAMESPACE "object_marker"
#define MIN_NUM_OBJECT_POINTS 10 //VERY TENTATIVE 10
#define RANSAC_MAX_ITERATIONS 10000
#define PLANE_DIST_TRESH 0.025 //maximum distance from plane
#define MIN_NUMBER_PLANE_POINTS 500 //Lower??
#define EPS_ANGLE 0.05 //epsilon angle for segmenting, value in radians
#define CLUSTER_TOLERANCE 0.03 //Tentatively set at 3 cm for object cluster tolerance
//#define MAXIMUM_OBJECT_HEIGHT 0.30 //1 foot max height
#define BUFFER_DISTANCE 0.08   //from the edges of the bounding box for the plane


using namespace std;


bool VillaSurfaceDetector::segment_objects(Eigen::Vector3f axis, const PointCloudT::Ptr &world_points,
  const geometry_msgs::Quaternion &plane_coefs,
  const visualization_msgs::Marker &plane_bounding_box,
  float maximum_height,
  vector<PointCloudT::Ptr> &objects_out,
  vector<visualization_msgs::Marker> &object_bounding_boxes_out) {

    // Extract the information of the plane to get the objects on top of from from the given bounding box
    Eigen::Vector3f rotation, translation;
    translation[0] = plane_bounding_box.pose.position.x;
    translation[1] = plane_bounding_box.pose.position.y;
    translation[2] = plane_bounding_box.pose.position.z;

    Eigen::Quaternionf quant(plane_bounding_box.pose.orientation.w, plane_bounding_box.pose.orientation.x\
    , plane_bounding_box.pose.orientation.y, plane_bounding_box.pose.orientation.z);
    rotation = quant.toRotationMatrix().eulerAngles(0, 1, 2);

    // Define the bounds of the crop box around the points we wish to isolate
    Eigen::Vector4f minPoint, maxPoint;
    maxPoint[0] = (plane_bounding_box.scale.x / 2) - BUFFER_DISTANCE;
    maxPoint[1] = (plane_bounding_box.scale.y / 2) - BUFFER_DISTANCE;
    maxPoint[2] = (plane_bounding_box.scale.z / 2) + maximum_height;
    minPoint[0] = (-1) * maxPoint[0];
    minPoint[1] = (-1) * maxPoint[1];
    minPoint[2] = (-1) * (plane_bounding_box.scale.z / 2);

    // Execute the crop box operation on the given cloud
    PointCloudT::Ptr isolated_cloud(new PointCloudT);
    pcl::CropBox<PointT> crop;
    crop.setMin(minPoint);
    crop.setMax(maxPoint);
    crop.setRotation(rotation);
    crop.setTranslation(translation);
    crop.setInputCloud(world_points);
    crop.filter(*isolated_cloud);

    // Create the segmentation object and target cloud for object extraction
  	pcl::SACSegmentation<PointT> seg;
  	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  	PointCloudT::Ptr objects_cloud(new PointCloudT);
  	seg.setOptimizeCoefficients (true);

  	// Look for a plane perpendicular to a given axis (z-axis)
  	seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
  	seg.setMethodType (pcl::SAC_RANSAC);
  	seg.setMaxIterations (RANSAC_MAX_ITERATIONS);
  	seg.setDistanceThreshold (PLANE_DIST_TRESH);
  	seg.setEpsAngle(EPS_ANGLE);
    //Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0);
  	seg.setAxis(axis);

    seg.setInputCloud (isolated_cloud);
  	seg.segment (*inliers, *coefficients);

    // Extract dominant plane from isolated_cloud
    pcl::ExtractIndices<PointT> extract;
  	extract.setInputCloud (isolated_cloud);
  	extract.setIndices (inliers);
  	extract.setNegative (true);
  	extract.filter (*objects_cloud);

    // Create the euclidean extraction tools for object extraction
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (objects_cloud);

    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (CLUSTER_TOLERANCE);
    ec.setMinClusterSize (MIN_NUM_OBJECT_POINTS);
    ec.setSearchMethod (tree);
    ec.setInputCloud (objects_cloud);
    ec.extract (cluster_indices);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it < cluster_indices.end(); it++) {
      // Push back each cloud into the return vector
      PointCloudT::Ptr cluster(new PointCloudT(*objects_cloud, (*it).indices));
  		objects_out.push_back(cluster);

      // Calculate and push back marker for each object
      const BoundingBox &oriented_bbox_params = extract_oriented_bounding_box_params(cluster);
      visualization_msgs::Marker object_bounding_box_marker = createBoundingBoxMarker(oriented_bbox_params, 0, TARGET_FRAME);
      object_bounding_boxes_out.push_back(object_bounding_box_marker);
  	}

    return true;
  }
