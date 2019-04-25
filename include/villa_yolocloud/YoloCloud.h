#pragma once

#include <string>
#include <iostream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <visualization_msgs/Marker.h>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <octomap/octomap.h>
#include <Eigen/Dense>
#include <nanoflann.hpp>

struct ImageBoundingBox {
    float x;
    float y;
    int width;
    int height;
    std::string label;
};

struct YoloCloudObject {
    Eigen::Vector3f position;
    std::string label;
    int id;

    bool invalid() {
        return label.empty();
    }
};


class YoloCloud {
private:
    // Hardcoded intrinsic parameters for the HSR xtion
    float intrinsic_sx = 535.2900990271;
    float intrinsic_sy = 535.2900990271;
    float intrinsic_cx = 320.0000000000;
    float intrinsic_cy = 240.0000000000;

    // Point Cloud definition for nanoflann
    struct PointCloud {
        struct Point {
            float x,y,z;
            int id;
        };

        int uuid = 0;  // Unique label for a point
        std::vector<Point> pts;

        inline void add(float x, float y, float z) {
            Point p;
            p.x = x;
            p.y = y;
            p.z = z;
            p.id = uuid++;
            pts.push_back(p);
        }

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts.size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            if (dim == 0) return pts[idx].x;
            else if (dim == 1) return pts[idx].y;
            else return pts[idx].z;
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

    };

    // construct a kd-tree index:
    typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
            nanoflann::L2_Simple_Adaptor<float, PointCloud>,
            PointCloud,
            3 /* dim */
    > KDTreeIndex;

    // Our object cloud
    PointCloud cloud;
    KDTreeIndex cloudIndex;

    std::unordered_map<int, YoloCloudObject> objectsData;

public:
    YoloCloud()
        : cloudIndex(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams()) {
    }

    std::pair<bool, YoloCloudObject> addObject(const ImageBoundingBox &bbox, const cv::Mat &rgb_image, const cv::Mat &depth_image, const Eigen::Affine3f &camToMap);

    std::vector<YoloCloudObject> getAllObjects();

    std::vector<YoloCloudObject> searchBox(Eigen::Vector3f min, Eigen::Vector3f max);

};
