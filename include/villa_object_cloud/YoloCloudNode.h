#pragma once

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <std_srvs/Empty.h>
#include <villa_object_cloud/ObjectCloud.h>
#include <villa_object_cloud/YoloModel.h>
#include <villa_object_cloud/BlockingQueue.h>
#include <villa_object_cloud/DetectedObject.h>
#include <villa_object_cloud/GetEntities.h>
#include <villa_object_cloud/GetBoundingBoxes.h>
#include <villa_object_cloud/GetObjects.h>
#include <villa_object_cloud/GetSurfaces.h>
#include <villa_object_cloud/GetSurfaceOccupancy.h>
#include <villa_object_cloud/ObjectCloudNode.h>
#include <knowledge_representation/LongTermMemoryConduit.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#define VISUALIZE 1
#define VISUALIZE_OCTREE 1

class YoloCloudNode : public ObjectCloudNode {
private:

    std::mutex yolo_mutex;  // Yolo lock


    // Yolo Publishers per camera
    std::unordered_map<std::string, ros::Publisher> yolo_pubs;



public:
    YoloModel model;

    explicit YoloCloudNode(ros::NodeHandle node);

    void data_callback(const sensor_msgs::Image::ConstPtr &rgb_image,
                       const sensor_msgs::Image::ConstPtr &depth_image,
                       const nav_msgs::Odometry::ConstPtr &odom);

    void hand_camera_callback(const sensor_msgs::Image::ConstPtr &rgb_image);

};
