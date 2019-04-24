#pragma once

#include <string>
#include <iostream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <octomap/octomap.h>
#include <Eigen/Dense>

struct ImageBoundingBox {
    float x;
    float y;
    int width;
    int height;
    std::string label;
};

struct YoloCloudObject {
    octomap::point3d position;
    std::string label;

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

    std::unordered_map<octomap::OcTreeKey, YoloCloudObject, octomap::OcTreeKey::KeyHash> objectsData;

public:
    octomap::OcTree octree;
    YoloCloud()
        : octree(0.01) {
    }

    std::pair<bool, YoloCloudObject> addObject(const ImageBoundingBox &bbox, const cv::Mat &rgb_image, const cv::Mat &depth_image, const Eigen::Affine3f &camToMap);

    std::vector<YoloCloudObject> getAllObjects();

};
