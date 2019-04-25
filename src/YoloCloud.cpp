#include "villa_yolocloud/YoloCloud.h"
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace cv;
using namespace nanoflann;
using namespace std;

const float NEIGHBOR_THRESH_X = 0.2;
const float NEIGHBOR_THRESH_Y = 0.2;
const float NEIGHBOR_THRESH_Z = 0.3;

/*
 * Adds object
 * Returns (new object?, 3D point)
 */
pair<bool, YoloCloudObject> YoloCloud::addObject(const ImageBoundingBox &bbox,
                                                    const Mat &rgb_image,
                                                    const Mat &depth_image,
                                                    const Eigen::Affine3f &camToMap) {
    if (bbox.y + bbox.height > rgb_image.rows
        || bbox.x + bbox.width > rgb_image.cols) {
        //std::cout << bbox.x << std::endl;
        //std::cout << bbox.y << std::endl;
        //std::cout << bbox.width << std::endl;
        //std::cout << bbox.height << std::endl;
        //throw std::runtime_error("Invalid bbox");
        return make_pair(false, YoloCloudObject());
    }

    // Get the depth of the object
    // We could use GrabCut, but it takes too long ~5fps
    // So just get the depth of the center pixel
    // It's probably good enough ¯\_(ツ)_/¯
    float depth = depth_image.at<uint16_t>(bbox.y + bbox.height/2., bbox.x + bbox.width/2.) / 1000.;
    if (depth == 0 || isnan(depth)) {
        return make_pair(false, YoloCloudObject());
    }

    // Compute 3d point in the world
    float x_center = bbox.x + bbox.width/2.;
    float y_center = bbox.y + bbox.height/2.;
    float camera_x = (x_center - intrinsic_cx) * (1.0 / intrinsic_sx) * depth;
    float camera_y = (y_center - intrinsic_cy) * (1.0 / intrinsic_sy) * depth;
    float camera_z = depth;
    Eigen::Vector3f world = camToMap * Eigen::Vector3f(camera_x, camera_y, camera_z);

    // Check if point exists in KDTree
    float candidatePoint[3] = { world(0), world(1), world(2) };

    // We want to search the smallest ball that contains our threshold box
    if (!cloud.pts.empty()) {
        float sqRadius = NEIGHBOR_THRESH_X * NEIGHBOR_THRESH_X + NEIGHBOR_THRESH_Y * NEIGHBOR_THRESH_Y +
                         NEIGHBOR_THRESH_Z * NEIGHBOR_THRESH_Z;
        std::vector<std::pair<size_t, float>> indices_dists;
        RadiusResultSet<float, size_t> resultSet(sqRadius, indices_dists);
        cloudIndex.findNeighbors(resultSet, candidatePoint, nanoflann::SearchParams());

        for (const auto &e : indices_dists) {
            PointCloud::Point &p = cloud.pts[e.first];
            if (std::abs(candidatePoint[0] - p.x) <= NEIGHBOR_THRESH_X
                && std::abs(candidatePoint[1] - p.y) <= NEIGHBOR_THRESH_Y
                && std::abs(candidatePoint[2] - p.z) <= NEIGHBOR_THRESH_Z) {

                // Same label?
                auto mit = objectsData.find(p.id);
                if (mit != objectsData.end() && mit->second.label == bbox.label) {
                    return make_pair(false, mit->second);
                }

            }
        }
    }

    // If it doesn't exist, we can add it to our cloud
    cloud.add(world(0), world(1), world(2));
    int idx = cloud.pts.size() - 1;
    cloudIndex.addPoints(idx, idx);  // [start, end] inclusive
    YoloCloudObject object;
    object.position = Eigen::Vector3f(world(0), world(1), world(2));
    object.label = bbox.label;
    object.id = cloud.pts[idx].id;
    objectsData.insert({cloud.pts[idx].id, object});

    return std::make_pair(true, object);
}

vector<YoloCloudObject> YoloCloud::getAllObjects() {
    vector<YoloCloudObject> objects;
    objects.reserve(objectsData.size());

    for (const auto &e : objectsData) {
        objects.push_back(e.second);
    }

    return objects;
}

vector<YoloCloudObject> YoloCloud::searchBox(Eigen::Vector3f min, Eigen::Vector3f max) {
    vector<YoloCloudObject> objects;

    // We want to search the smallest ball that contains our threshold box
    float sqRadius = (max(0)-min(0))*(max(0)-min(0)) + (max(1)-min(1))*(max(1)-min(1)) + (max(2)-min(2))*(max(2)-min(2));
    float center[3] = { (min(0) + max(0))/2.0f, (min(1) + max(1))/2.0f, (min(2) + max(2))/2.0f };
    std::vector<std::pair<size_t, float>> indices_dists;
    RadiusResultSet<float, size_t> resultSet(sqRadius, indices_dists);
    cloudIndex.findNeighbors(resultSet, center, nanoflann::SearchParams());

    for (const auto &e : indices_dists) {
        PointCloud::Point &p = cloud.pts[e.first];
        if (std::abs(center[0] - p.x) <= (max(0)-min(0))
            && std::abs(center[1] - p.y) <= (max(1)-min(1))
            && std::abs(center[2] - p.z) <= (max(2)-min(2))) {

            auto mit = objectsData.find(p.id);
            if (mit != objectsData.end()) {
                objects.push_back(mit->second);
            }

        }
    }

    return objects;
}
