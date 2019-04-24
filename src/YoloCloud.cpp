#include "villa_yolocloud/YoloCloud.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>

using namespace cv;
using namespace std;

const float NEIGHBOR_THRESH = 0.1;

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

    // Check if point exists in octomap
    octomap::point3d point(0, 0, 0);
    octomap::point3d min(world(0) - NEIGHBOR_THRESH, world(1) - NEIGHBOR_THRESH, world(2) - NEIGHBOR_THRESH);
    octomap::point3d max(world(0) + NEIGHBOR_THRESH, world(1) + NEIGHBOR_THRESH, world(2) + NEIGHBOR_THRESH);
    for (octomap::OcTree::leaf_bbx_iterator iter = octree.begin_leafs_bbx(min, max),
             end=octree.end_leafs_bbx(); iter != end; ++iter) {
        if (iter->getOccupancy() >= octree.getOccupancyThres()) {
            auto p = objectsData.find(iter.getKey());

            if (p != objectsData.end()) {

                // Is this point already marked as belonging to our object?
                if (p->second.label == bbox.label) {

                    // We don't want to add this as it is probably a duplicate
                    return make_pair(false, p->second);
                }

            } else {
                point = iter.getCoordinate();
            }
        }
    }

    // If point was never set then return
    if (point == octomap::point3d(0, 0, 0)) {
        return std::make_pair(false, YoloCloudObject());
    }

    // If it doesn't exist, we can add it to our dictionary
    octomap::OcTreeKey key = octree.coordToKey(point);
    YoloCloudObject object;
    object.position = point;
    object.label = bbox.label;
    objectsData.insert({key, object});

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