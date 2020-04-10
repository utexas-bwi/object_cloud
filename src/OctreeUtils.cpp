#include "villa_yolocloud/OctreeUtils.h"
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

using namespace cv;
using namespace std;

visualization_msgs::Marker OctreeUtils::extractBoundingBox(
    const octomap::OcTree &octree, const octomap::point3d &point,
    const cv::Rect &yolobbox, const Eigen::Affine3f &camToMap,
    const Eigen::Matrix3f &ir_intrinsics) {
  // This gets the min and max of the rough bounding box to look for object
  // points
  // The min and max is cutoff by the true min and max of our Octomap
  double min_bb_x;
  double min_bb_y;
  double min_bb_z;
  double max_bb_x;
  double max_bb_y;
  double max_bb_z;
  octree.getMetricMin(min_bb_x, min_bb_y, min_bb_z);
  octree.getMetricMax(max_bb_x, max_bb_y, max_bb_z);
  min_bb_x = std::max(min_bb_x, point.x() - 0.2);
  min_bb_y = std::max(min_bb_y, point.y() - 0.2);
  min_bb_z = std::max(min_bb_z, point.z() - 0.3);
  max_bb_x = std::min(max_bb_x, point.x() + 0.2);
  max_bb_y = std::min(max_bb_y, point.y() + 0.2);
  max_bb_z = std::min(max_bb_z, point.z() + 0.3);
  octomap::point3d min(min_bb_x, min_bb_y, min_bb_z);
  octomap::point3d max(max_bb_x, max_bb_y, max_bb_z);

  // Octomap stores free space, so just focus on occupied cells within our rough
  // bounding box
  std::vector<Eigen::Vector3f> pts;
  float minZ = std::numeric_limits<float>::max();
  for (octomap::OcTree::leaf_bbx_iterator
           iter = octree.begin_leafs_bbx(min, max),
           end = octree.end_leafs_bbx();
       iter != end; ++iter) {
    if (iter->getOccupancy() >= octree.getOccupancyThres()) {
      pts.emplace_back(iter.getCoordinate().x(), iter.getCoordinate().y(),
                       iter.getCoordinate().z());
      minZ = std::min(minZ, iter.getCoordinate().z());
    }
  }

  // Filter out z's that are too low (e.g. if part of the table is captured)
  std::vector<Eigen::Vector3f> ptsZFiltered;
  for (const auto &pt : pts) {
    if (pt(2) > minZ + 0.045) {
      ptsZFiltered.push_back(pt);
    }
  }
  pts = ptsZFiltered;

  // Prepare batch of points to check if in YOLO bounding box
  Eigen::Matrix<float, 3, Eigen::Dynamic> candidatePoints(3, pts.size());
  for (int i = 0; i < pts.size(); ++i) {
    candidatePoints.col(i) = pts[i];
  }

  // Project batch
  Eigen::Affine3f mapToCam = camToMap.inverse();
  Eigen::Matrix<float, 3, Eigen::Dynamic> imagePoints =
      ir_intrinsics * mapToCam * candidatePoints;
  imagePoints.array().rowwise() /= imagePoints.row(2).array();

  // Construct PCL Point Cloud with subset of points that project into YOLO
  // bounding box
  pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < pts.size(); ++i) {
    float x = imagePoints(0, i);
    float y = imagePoints(1, i);

    int x_b = yolobbox.x;
    int y_b = yolobbox.y;

    // Within YOLO Box
    if (x_b <= x && x < x_b + yolobbox.width && y_b <= y &&
        y < y_b + yolobbox.height) {
      objectCloud->points.emplace_back(pts[i](0), pts[i](1), pts[i](2));
    }
  }

  if (objectCloud->points.empty()) {
    return visualization_msgs::Marker();
  }

  // Use Euclidean clustering to filter out noise like objects that are behind
  // our object
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(objectCloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(0.02); // 2cm
  ec.setMinClusterSize(50);
  ec.setMaxClusterSize(25000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(objectCloud);
  ec.extract(cluster_indices);

  if (cluster_indices.empty()) {
    return visualization_msgs::Marker();
  }

  // TODO: Closest cluster might be better than max...
  auto max_cluster = std::max_element(
      cluster_indices.begin(), cluster_indices.end(),
      [](const pcl::PointIndices &a, const pcl::PointIndices &b) {
        return a.indices.size() < b.indices.size();
      });

  pcl::PointCloud<pcl::PointXYZ>::Ptr clusteredCloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (std::vector<int>::const_iterator pit = max_cluster->indices.begin();
       pit != max_cluster->indices.end(); ++pit) {
    pcl::PointXYZ pt = objectCloud->points[*pit];
    clusteredCloud->points.push_back(pt);
  }

  // The following gets and computes the oriented bounding box

  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
  feature_extractor.setInputCloud(clusteredCloud);
  feature_extractor.compute();

  pcl::PointXYZ min_point_OBB;
  pcl::PointXYZ max_point_OBB;
  pcl::PointXYZ position_OBB;
  Eigen::Matrix3f rotation_matrix_OBB;
  Eigen::Vector3f major_vector, middle_vector, minor_vector;

  feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB,
                           rotation_matrix_OBB);
  feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);

  // The orientation returned from getOBB is not what we want
  // There are no constraints, and it rearranges the axes in an arbitrary way
  // Instead, we will compute the rotation ourselves
  // from the eigenvectors of the (covariance of) the object's point cloud

  Eigen::Vector3f eigenvectors[3] = {major_vector, middle_vector, minor_vector};

  // Find the closest eigenvector to each of our principal axes
  // Since we're in a Hilbert space, the inner product tells us this
  Eigen::Vector3f x = Eigen::Vector3f::UnitX();
  Eigen::Vector3f y = Eigen::Vector3f::UnitY();
  Eigen::Vector3f z = Eigen::Vector3f::UnitZ();
  float cos_with_x[3] = {std::abs(x.dot(major_vector)),
                         std::abs(x.dot(middle_vector)),
                         std::abs(x.dot(minor_vector))};
  float cos_with_y[3] = {std::abs(y.dot(major_vector)),
                         std::abs(y.dot(middle_vector)),
                         std::abs(y.dot(minor_vector))};
  float cos_with_z[3] = {std::abs(z.dot(major_vector)),
                         std::abs(z.dot(middle_vector)),
                         std::abs(z.dot(minor_vector))};
  auto x_idx = std::max_element(cos_with_x, cos_with_x + 3) - cos_with_x;
  auto y_idx = std::max_element(cos_with_y, cos_with_y + 3) - cos_with_y;
  auto z_idx = std::max_element(cos_with_z, cos_with_z + 3) - cos_with_z;

  // I now want to rotate our x axis to a "grounded" version
  // of the eigenvector closest to the x axis
  // By grounded I mean projected onto the ground
  // In other words, we're only concerned with a change in yaw
  Eigen::Vector3f grounded_axis(eigenvectors[x_idx]);
  grounded_axis(2) = 0.;
  grounded_axis.normalize();
  Eigen::Quaternionf rotation;
  rotation = Eigen::AngleAxisf(acos(x.dot(grounded_axis)),
                               x.cross(grounded_axis).normalized());

  // getOBB returns bounds for the box, where the first component has the major
  // second the middle, and third the minor axis bounds
  // We want to rearrange this such that the first component is x, second y, and
  // third z
  Eigen::Vector3f orig_min_point_OBB(min_point_OBB.x, min_point_OBB.y,
                                     min_point_OBB.z);
  Eigen::Vector3f orig_max_point_OBB(max_point_OBB.x, max_point_OBB.y,
                                     max_point_OBB.z);
  Eigen::Vector3f min_point(orig_min_point_OBB(x_idx),
                            orig_min_point_OBB(y_idx),
                            orig_min_point_OBB(z_idx));
  Eigen::Vector3f max_point(orig_max_point_OBB(x_idx),
                            orig_max_point_OBB(y_idx),
                            orig_max_point_OBB(z_idx));

  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.id = 1;
  marker.type = 1;
  marker.action = 0;
  marker.pose.position.x = position_OBB.x;
  marker.pose.position.y = position_OBB.y;
  marker.pose.position.z = position_OBB.z;
  marker.pose.orientation.x = rotation.x();
  marker.pose.orientation.y = rotation.y();
  marker.pose.orientation.z = rotation.z();
  marker.pose.orientation.w = rotation.w();
  marker.scale.x = max_point(0) - min_point(0);
  marker.scale.y = max_point(1) - min_point(1);
  marker.scale.z = max_point(2) - min_point(2);
  marker.color.r = 0.;
  marker.color.b = 0.;
  marker.color.g = 1.;
  marker.color.a = 1.;
  marker.lifetime = ros::Duration(0);

  return marker;
}
