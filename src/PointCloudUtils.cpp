#include <object_cloud/PointCloudUtils.h>
#include <chrono>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <object_cloud/SACModelEdge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/impl/ransac.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2_eigen/tf2_eigen.h>

// How much extra height to give to surfaces (as a safety)
#define SURFACE_EXTRA_Z 0.03

// How close does an object have to be to count as being on a surface?
#define SURFACE_CLOSENESS 0.2

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace cv;
using namespace std;

visualization_msgs::Marker PointCloudUtils::extractBoundingBox(const octomap::OcTree &octree,
                                                               const octomap::point3d &point,
                                                               const cv::Rect &yolobbox,
                                                               const Eigen::Affine3f &camToMap,
                                                               const Eigen::Affine3f &baseToMap,
                                                               const Eigen::Matrix3f &ir_intrinsics) {
    // This gets the min and max of the rough bounding box to look for object points
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

    // Octomap stores free space, so just focus on occupied cells within our rough bounding box
    std::vector<Eigen::Vector3f> pts;
    float minZ = std::numeric_limits<float>::max();
    for (octomap::OcTree::leaf_bbx_iterator iter = octree.begin_leafs_bbx(min, max),
                 end = octree.end_leafs_bbx(); iter != end; ++iter) {
        if (iter->getOccupancy() >= octree.getOccupancyThres()) {
            pts.emplace_back(iter.getCoordinate().x(), iter.getCoordinate().y(), iter.getCoordinate().z());
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
    Eigen::Matrix<float, 3, Eigen::Dynamic> imagePoints = ir_intrinsics * mapToCam * candidatePoints;
    imagePoints.array().rowwise() /= imagePoints.row(2).array();

    // Construct PCL Point Cloud with subset of points that project into YOLO bounding box
    pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < pts.size(); ++i) {
        float x = imagePoints(0, i);
        float y = imagePoints(1, i);

        int x_b = yolobbox.x;
        int y_b = yolobbox.y;

        // Within YOLO Box
        if (x_b <= x && x < x_b + yolobbox.width && y_b <= y && y < y_b + yolobbox.height) {
            objectCloud->points.emplace_back(pts[i](0), pts[i](1), pts[i](2));
        }
    }

    if (objectCloud->points.empty()) {
        return visualization_msgs::Marker();
    }

    // Use Euclidean clustering to filter out noise like objects that are behind our object
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
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

    // Pick Closest Cluster
    Eigen::Affine3f mapToBase = baseToMap.inverse();
    float closest_dist = numeric_limits<float>::max();
    int closest_cluster_idx = 0;
    for (int cluster_idx = 0; cluster_idx < cluster_indices.size(); cluster_idx++) {
        // Extract centroid
        pcl::CentroidPoint<pcl::PointXYZ> centroid;
        for (int i : cluster_indices[cluster_idx].indices) {
            centroid.add(objectCloud->points[i]);
        }
        pcl::PointXYZ c;
        centroid.get(c);

        Eigen::Vector3f baseCentroid = mapToBase * Eigen::Vector3f(c.x, c.y, c.z);
        float distance = Eigen::Vector2f(baseCentroid(0), baseCentroid(1)).norm();
        if (distance < closest_dist) {
            closest_dist = distance;
            closest_cluster_idx = cluster_idx;
        }
    }

    const pcl::PointIndices &max_cluster = cluster_indices[closest_cluster_idx];

    pcl::PointCloud<pcl::PointXYZ>::Ptr clusteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = max_cluster.indices.begin();
         pit != max_cluster.indices.end(); ++pit) {
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

    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);
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
    float cos_with_x[3] = {
            std::abs(x.dot(major_vector)),
            std::abs(x.dot(middle_vector)),
            std::abs(x.dot(minor_vector))
    };
    float cos_with_y[3] = {
            std::abs(y.dot(major_vector)),
            std::abs(y.dot(middle_vector)),
            std::abs(y.dot(minor_vector))
    };
    float cos_with_z[3] = {
            std::abs(z.dot(major_vector)),
            std::abs(z.dot(middle_vector)),
            std::abs(z.dot(minor_vector))
    };
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
    rotation = Eigen::AngleAxisf(acos(x.dot(grounded_axis)), x.cross(grounded_axis).normalized());

    // getOBB returns bounds for the box, where the first component has the major
    // second the middle, and third the minor axis bounds
    // We want to rearrange this such that the first component is x, second y, and third z
    Eigen::Vector3f orig_min_point_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
    Eigen::Vector3f orig_max_point_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
    Eigen::Vector3f min_point(orig_min_point_OBB(x_idx), orig_min_point_OBB(y_idx), orig_min_point_OBB(z_idx));
    Eigen::Vector3f max_point(orig_max_point_OBB(x_idx), orig_max_point_OBB(y_idx), orig_max_point_OBB(z_idx));

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
    marker.scale.x = (max_point(0) - min_point(0)) - octree.getResolution();
    marker.scale.y = (max_point(1) - min_point(1)) - octree.getResolution();
    marker.scale.z = max_point(2) - min_point(2);
    marker.color.r = 0.;
    marker.color.b = 0.;
    marker.color.g = 1.;
    marker.color.a = 1.;
    marker.lifetime = ros::Duration(0);

    return marker;
}


/**
 * Extracts surface (e.g. table, countertop)
 * @param cloud Point cloud in the straightCam frame (i.e. rotated base_link, where origin ground, z-up, x-forward)
 * @return
 */
boost::optional<visualization_msgs::Marker> PointCloudUtils::extractSurface(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const Eigen::Affine3f &camToMap,
        const Eigen::Affine3f &camToStraightCam,
        float xScale,
        float yScale,
        float zScale,
        ros::Publisher &publisher) {
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Return if cloud has nothing
    if (cloud->points.empty()) {
        std::cout << "Empty Cloud!" << std::endl;
        return boost::optional<visualization_msgs::Marker>();
    }

    std::vector<Eigen::Vector3f> invalids = {Eigen::Vector3f::UnitX()};

    std::vector<Eigen::VectorXf> linecoeffs;
    std::vector<std::vector<int>> line_indices;
    int best = -1;
    float bestAvg = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; i++) {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

        // Create SAC model
        SACModelEdge<PointT>::Ptr sacmodel(new SACModelEdge<PointT>(cloud,
                                                                    invalids,
                                                                    0.3,  // min length in meters of edge
                                                                    2));  // min density
        sacmodel->setAxis(Eigen::Vector3f(0., 0., 1.));
        sacmodel->setEpsCosine(0.03);

        // Create SAC method
        pcl::RandomSampleConsensus<PointT>::Ptr sac(new pcl::RandomSampleConsensus<PointT>(sacmodel));
        sac->setProbability(0.99);
        sac->setDistanceThreshold(0.01);
        sac->setMaxIterations(100000);

        // Fit model
        if (!sac->computeModel()) {
            ROS_WARN_STREAM("SAC could not compute model!");
            return boost::optional<visualization_msgs::Marker>();
        }

        // Get inliers
        sac->getInliers(inliers->indices);

        // Get the model coefficients
        Eigen::VectorXf coeff;
        sac->getModelCoefficients(coeff);

        // Optimized coefficients
        Eigen::VectorXf coeff_refined;
        sacmodel->optimizeModelCoefficients(inliers->indices, coeff, coeff_refined);
        coeff = coeff_refined;
        // Refine inliers
        sacmodel->selectWithinDistance(coeff_refined, sac->getDistanceThreshold(), inliers->indices);

        if (inliers->indices.empty()) {
            break;
        }

        line_indices.push_back(inliers->indices);
        linecoeffs.push_back(coeff);

        float avgX = 0;
        for (int idx : inliers->indices) {
            avgX += cloud->points[idx].x;

        }
        avgX /= inliers->indices.size();

        if (avgX < bestAvg) {
            best = i;
            bestAvg = avgX;
        }

        invalids.emplace_back(coeff[3], coeff[4], coeff[5]);
    }

    if (best < 0) {
        std::cout << "Failed to get surface!" << std::endl;
        return boost::optional<visualization_msgs::Marker>();
    }

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    inliers->indices = line_indices[best];

    // Create the filtering object
    pcl::ExtractIndices<PointT> extract;

    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    // Visualize
    PointCloudT::Ptr plane(new PointCloudT);
    extract.filter(*plane);
    PointCloudT::Ptr planecp(new PointCloudT);
    pcl::transformPointCloud(*plane, *planecp, camToMap * camToStraightCam.inverse());
    planecp->header.frame_id = "map";
    publisher.publish(planecp);

    auto &coeffs = linecoeffs[best];
    Eigen::Vector3f linedir(coeffs(3), coeffs(4), 0);
    linedir.normalize();
    float angle = std::acos(Eigen::Vector3f::UnitY().dot(linedir));
    Eigen::Affine3f lineToStraightCam;
    lineToStraightCam = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitY().cross(linedir).normalized());

    // Project origin onto line and make that the centré
    // This isn't strictly necessary, but it makes it more stable
    Eigen::Vector2f slope(coeffs(3), coeffs(4));
    Eigen::Vector2f center = Eigen::Vector2f(coeffs(0), coeffs(1)) +
                             ((-Eigen::Vector2f(coeffs(0), coeffs(1))).dot(slope) / slope.dot(slope)) * slope;

    lineToStraightCam = Eigen::Translation3f(center(0), center(1), coeffs(2) - zScale / 2) * lineToStraightCam;
    lineToStraightCam *= Eigen::Translation3f(-xScale / 2, 0, 0);  // Translate to the center of the box

    Eigen::Affine3f lineToMap = camToMap * camToStraightCam.inverse() * lineToStraightCam;

    Eigen::Quaternionf orient;
    orient = lineToMap.rotation();
    Eigen::Vector3f pos = lineToMap.translation();

    visualization_msgs::Marker surface;
    surface.header.frame_id = "map";
    surface.type = 1;
    surface.action = 0;
    surface.pose.position.x = pos(0);
    surface.pose.position.y = pos(1);
    surface.pose.position.z = pos(2);
    surface.pose.orientation.x = orient.x();
    surface.pose.orientation.y = orient.y();
    surface.pose.orientation.z = orient.z();
    surface.pose.orientation.w = orient.w();
    surface.scale.x = xScale;
    surface.scale.y = yScale;
    surface.scale.z = zScale + SURFACE_EXTRA_Z;
    surface.color.r = 1.;
    surface.color.b = 1.;
    surface.color.g = 1.;
    surface.color.a = 1.;
    surface.lifetime = ros::Duration(0);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = finish - start;
    std::cout << "\tRANSAC Elapsed time: " << elapsed2.count() << " s\n";

    return boost::optional<visualization_msgs::Marker>(surface);
}

vector<Eigen::Vector2f> PointCloudUtils::surfaceOccupancy(vector<visualization_msgs::Marker> all_objects,
                                                          visualization_msgs::Marker surface,
                                                          Eigen::Vector2f xbounds,
                                                          Eigen::Vector2f ybounds) {

    Eigen::Affine3d surfaceToGround;
    tf2::fromMsg(surface.pose, surfaceToGround);

    vector<Eigen::Vector2f> obstacles;
    for (const auto &obj : all_objects) {
        // Sanity check, ground should always be the same by default
        if (obj.header.frame_id != surface.header.frame_id) {
            throw runtime_error("Mismatched ground frames!");
        }

        Eigen::Affine3d objToGround;
        tf2::fromMsg(obj.pose, objToGround);
        Eigen::Affine3f objToSurface = (surfaceToGround.inverse() * objToGround).cast<float>();

        Eigen::Vector3f center = objToSurface * Eigen::Vector3f(0, 0, -obj.scale.z / 2);

        // If this object is on a different surface then ignore
        if (std::abs(center(2)) > SURFACE_CLOSENESS ||
            center(0) < xbounds(0) || center(0) > xbounds(1) ||
            center(1) < ybounds(0) || center(1) > ybounds(1)) {
            continue;
        }

        // Four corners of rectangle in object frame
        vector<Eigen::Vector3f> corners;
        corners.emplace_back(-obj.scale.x / 2, obj.scale.y / 2, obj.scale.z / 2);
        corners.emplace_back(-obj.scale.x / 2, -obj.scale.y / 2, obj.scale.z / 2);
        corners.emplace_back(obj.scale.x / 2, -obj.scale.y / 2, obj.scale.z / 2);
        corners.emplace_back(obj.scale.x / 2, obj.scale.y / 2, obj.scale.z / 2);

        // Get corners in surface frame
        for (auto &corn : corners) {
            corn = objToSurface * corn;
        }

        float minY = std::min({corners[0](1), corners[1](1), corners[2](1), corners[3](1)});
        float maxY = std::max({corners[0](1), corners[1](1), corners[2](1), corners[3](1)});

        // Push the y span of the object
        obstacles.emplace_back(minY, maxY);
    }

    return obstacles;
}
