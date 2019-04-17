#ifndef DETECTIONMODEL_H
#define DETECTIONMODEL_H

#include <string>
#include <iostream>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "darknet.h"

extern "C" image mat_to_image(cv::Mat m);
extern "C" network *parse_network_cfg(char *filename);

struct Detection {
    std::string label;
    cv::Rect bbox;
};

/*
 * Provides predictions from YOLO network
 */
class YoloModel {
private:
    float kThresh = 0.5;
    float kHierThresh = 0.5;
    float kNMS = 0.45; // Non-maximum suppression threshold

    // Darknet image objects
    // For some reason we can't instantiate darknet
    // images in the cpp file...
    // So just have a image array, two since one can be a temp image
    image im_[2];

    // List of class labels
    std::vector<std::string> labels_;

    // YOLO Network
    // We don't want to use unique_ptr
    // as the free_network function needs to be called
    network *net_;

public:
    void load(std::string labels_file, std::string model_config_file, std::string weights_file);
    // Unsafe pointer due to darknet...
    std::vector<Detection> detect(cv::Mat *image);
    void unload();

    ~YoloModel();
};

#endif // DETECTIONMODEL_H
