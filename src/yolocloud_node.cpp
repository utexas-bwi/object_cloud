#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

#include <sensor_msgs/Image.h>
#include <tmc_yolo2_ros/Detections.h>
#include <tmc_yolo2_ros/Detection.h>

#include <knowledge_representation/LongTermMemoryConduit.h>
#include <villa_yolocloud/YoloCloud.h>
#include <villa_yolocloud/DetectedObject.h>
#include <villa_yolocloud/GetShelfObjects.h>
#include <villa_yolocloud/GetEntities.h>

#define VISUALIZE true

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        tmc_yolo2_ros::Detections> SyncPolicy;


class YoloCloudNode {
private:
    YoloCloud shelf_manager;
    tf::TransformListener listener;
    std::unordered_map<int, int> entity_id_to_point;
    knowledge_rep::LongTermMemoryConduit ltmc; // Knowledge base

    ros::Publisher viz_pub;

public:
    YoloCloudNode(ros::NodeHandle node)
        : ltmc("127.0.0.1", 33060, "root", "", "villa_krr") {
        viz_pub = node.advertise<visualization_msgs::MarkerArray>("yoloobjects/markers", 1, true);
    }

    void data_callback(const sensor_msgs::Image::ConstPtr &rgb_image,
                       const sensor_msgs::Image::ConstPtr &depth_image,
                       const tmc_yolo2_ros::Detections::ConstPtr &yolo_detections) {
        if (yolo_detections->detections.size() == 0) {
            return;
        }

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        tf::StampedTransform transform;
        try {
            listener.waitForTransform("map", "head_rgbd_sensor_link", rgb_image->header.stamp, ros::Duration(0.1));
            listener.lookupTransform("map", "head_rgbd_sensor_link", rgb_image->header.stamp, transform);
        } catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            return;
        }

        Eigen::Affine3d camToMap;
        tf::transformTFToEigen(transform, camToMap);

        cv::Mat depthI(depth_image->height, depth_image->width, CV_32FC1);
        memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());

        for (const tmc_yolo2_ros::Detection &detection : yolo_detections->detections) {
            ImageBoundingBox bbox;
            bbox.x = detection.x - detection.width/2;
            bbox.y = detection.y - detection.height/2;
            bbox.width = detection.width;
            bbox.height = detection.height;
            bbox.label = detection.class_name;

            int idx = shelf_manager.addObject(bbox, cv_ptr->image, depthI, camToMap.cast<float>());

            // If object was added to yolocloud, add to knowledge base
            if (idx >= 0) {
                add_to_ltmc(idx);
            }
        }

#ifdef VISUALIZE
        visualize();
#endif
    }


    void add_to_ltmc(int cloud_idx) {
        std::string label = shelf_manager.labels.at(shelf_manager.objects->points[cloud_idx].label);
        int cid = ltmc.get_concept(label);

        int eid = ltmc.add_entity();
        ltmc.add_entity_attribute(eid, "is_a", cid);
        ltmc.add_entity_attribute(eid, "sensed", true);
        entity_id_to_point.insert({eid, cloud_idx});
    }


    bool get_entities(villa_yolocloud::GetEntities::Request &req, villa_yolocloud::GetEntities::Response &res) {
        std::vector<geometry_msgs::Point> map_locations;
        for (int eid : req.entity_ids) {
            pcl::PointXYZL point = shelf_manager.objects->points[entity_id_to_point.at(eid)];

            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            map_locations.push_back(p);
        }

        res.map_locations = map_locations;

        return true;
    }


    bool get_shelf_objects(villa_yolocloud::GetShelfObjects::Request &req, villa_yolocloud::GetShelfObjects::Response &res) {
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = shelf_manager.sliceCloud(req.x_min, req.x_max, req.y_min, req.y_max, req.z_min, req.z_max);

        for (const auto &point : cloud->points) {
            villa_yolocloud::DetectedObject obj;
            obj.x = point.x;
            obj.y = point.y;
            obj.z = point.z;
            obj.label = shelf_manager.labels.at(point.label);
            res.objects.push_back(obj);
        }

        return true;
    }


    void visualize() {
        visualization_msgs::MarkerArray yolo_marker_array;
        for (const auto &object : shelf_manager.objects->points) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.ns = shelf_manager.labels.at(object.label);
            marker.id = 0;
            marker.type = 1;
            marker.action = 0;
            marker.pose.position.x = object.x;
            marker.pose.position.y = object.y;
            marker.pose.position.z = object.z;
            marker.pose.orientation.x = 0.;
            marker.pose.orientation.y = 0.;
            marker.pose.orientation.z = 0.;
            marker.pose.orientation.w = 1.;
            marker.scale.x = 0.10;
            marker.scale.y = 0.10;
            marker.scale.z = 0.10;
            marker.color.r = 1.;
            marker.color.b = 0.;
            marker.color.g = 0.;
            marker.color.a = 1.;
            marker.lifetime = ros::Duration(0);

            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = "map";
            text_marker.ns = shelf_manager.labels.at(object.label);
            text_marker.id = 1;
            text_marker.type = 9;
            text_marker.text = shelf_manager.labels.at(object.label);
            text_marker.action = 0;
            text_marker.pose.position.x = object.x;
            text_marker.pose.position.y = object.y;
            text_marker.pose.position.z = object.z + 0.10;
            text_marker.pose.orientation.x = 0.;
            text_marker.pose.orientation.y = 0.;
            text_marker.pose.orientation.z = 0.;
            text_marker.pose.orientation.w = 1.;
            text_marker.scale.x = 0.10;
            text_marker.scale.y = 0.10;
            text_marker.scale.z = 0.10;
            text_marker.color.r = 1.;
            text_marker.color.b = 0.;
            text_marker.color.g = 0.;
            text_marker.color.a = 1.;
            text_marker.lifetime = ros::Duration(0);

            yolo_marker_array.markers.push_back(marker);
            yolo_marker_array.markers.push_back(text_marker);
        }
        viz_pub.publish(yolo_marker_array);
    }
};

int main (int argc, char **argv) {
    ROS_INFO("Initializing yolocloud node...");
    ros::init(argc, argv, "yolocloud_node");
    ros::NodeHandle n;

    YoloCloudNode shelf(n);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/hsrb/head_rgbd_sensor/rgb/image_rect_color", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n, "/hsrb/head_rgbd_sensor/depth_registered/image", 10);
    message_filters::Subscriber<tmc_yolo2_ros::Detections> yolo_sub(n, "/yolo2_node/detections", 10);
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), image_sub, depth_sub, yolo_sub);
    sync.registerCallback(boost::bind(&YoloCloudNode::data_callback, &shelf, _1, _2, _3));

    ROS_INFO("Started. Waiting for inputs.");

    ros::ServiceServer getShelfObjects = n.advertiseService("getShelfObjects", &YoloCloudNode::get_shelf_objects, &shelf);
    ros::ServiceServer getEntities = n.advertiseService("getEntities", &YoloCloudNode::get_entities, &shelf);

    ros::spin();

    return 0;
}
