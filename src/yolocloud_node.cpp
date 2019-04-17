#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <knowledge_representation/LongTermMemoryConduit.h>
#include <knowledge_representation/convenience.h>
#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <knowledge_representation/LTMCInstance.h>
#include <villa_yolocloud/YoloCloud.h>
#include <villa_yolocloud/DetectedObject.h>
#include <villa_yolocloud/GetShelfObjects.h>
#include <villa_yolocloud/GetEntities.h>
#include <villa_yolocloud/YoloModel.h>
#include <opencv/highgui.h>

#define VISUALIZE true

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image> SyncPolicy;


class YoloCloudNode {
private:
    YoloCloud yoloCloud;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;
    std::unordered_map<int, int> entity_id_to_point;
    knowledge_rep::LongTermMemoryConduit ltmc; // Knowledge base


#ifdef VISUALIZE
    image_transport::Publisher viz_yolo_pub;
    ros::Publisher viz_pub;
#endif

public:
    image_transport::ImageTransport it;
    YoloModel model;
    bool received_first_message = false;

    YoloCloudNode(ros::NodeHandle node)
        : tfListener(tfBuffer),
          it(node),
          ltmc(knowledge_rep::get_default_ltmc()) {

#ifdef VISUALIZE
        viz_yolo_pub = it.advertise("yolocloud/detections", 1);
        viz_pub = node.advertise<visualization_msgs::MarkerArray>("yolocloud/markers", 1, true);
#endif

        // NOTE: We assume there's only one YoloCloud, so this will blow away anything that is sensed
        ltmc.get_concept("sensed").remove_instances();
        ltmc.get_concept("scanned").remove_references();
    }

    ~YoloCloudNode() {
        ltmc.get_concept("sensed").remove_instances();
        ltmc.get_concept("scanned").remove_references();
    }

    void data_callback(const sensor_msgs::Image::ConstPtr &rgb_image,
                       const sensor_msgs::Image::ConstPtr &depth_image) {
        received_first_message = true;
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        geometry_msgs::TransformStamped transform;
        try {
            transform = tfBuffer.lookupTransform("map", "head_rgbd_sensor_link", rgb_image->header.stamp, ros::Duration(0.01));
        } catch (tf2::TransformException &ex){
            ROS_ERROR("%s", ex.what());
            return;
        }

        Eigen::Affine3d camToMap = tf2::transformToEigen(transform);

        cv::Mat depthI(depth_image->height, depth_image->width, CV_32FC1);
        memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());

        std::vector<Detection> dets = model.detect(&cv_ptr->image);

        // Quit if there are no YOLO detections
        if (dets.size() == 0) {
            return;
        }

        for (const auto &detection : dets) {
            ImageBoundingBox bbox;
            bbox.x = detection.bbox.x;
            bbox.y = detection.bbox.y;
            bbox.width = detection.bbox.width;
            bbox.height = detection.bbox.height;
            bbox.label = detection.label;

            int idx = yoloCloud.addObject(bbox, cv_ptr->image, depthI, camToMap.cast<float>());

            // If object was added to yolocloud, add to knowledge base
            if (idx >= 0) {
                add_to_ltmc(idx);
            }
        }

#ifdef VISUALIZE
        for (const auto &d : dets) {
            cv::rectangle(cv_ptr->image, d.bbox, cv::Scalar(100, 100, 0), 3);
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
        viz_yolo_pub.publish(msg);

        visualize();
#endif
    }


    void add_to_ltmc(int cloud_idx) {
        std::string label = yoloCloud.labels.at(yoloCloud.objects->points[cloud_idx].label);
        auto concept = ltmc.get_concept(label);
        auto entity = concept.create_instance();
        auto sensed = ltmc.get_concept("sensed");
        entity.make_instance_of(sensed);
        entity_id_to_point.insert({entity.entity_id, cloud_idx});
    }


    bool get_entities(villa_yolocloud::GetEntities::Request &req, villa_yolocloud::GetEntities::Response &res) {
        std::vector<geometry_msgs::Point> map_locations;
        for (int eid : req.entity_ids) {
            auto points = yoloCloud.objects->points;
            geometry_msgs::Point p;
            // Requested an ID that's not in the cloud. Return NANs
            if (entity_id_to_point.count(eid) == 0) {
                knowledge_rep::Entity entity = {eid, ltmc};
                entity.remove_attribute("sensed");
                p.x = NAN;
                p.y = NAN;
                p.z = NAN;
            } else {
                pcl::PointXYZL point = yoloCloud.objects->points[entity_id_to_point.at(eid)];
                p.x = point.x;
                p.y = point.y;
                p.z = point.z;
            }

            map_locations.push_back(p);
        }

        res.map_locations = map_locations;

        return true;
    }


    void visualize() {
        visualization_msgs::MarkerArray yolo_marker_array;
        for (const auto &object : yoloCloud.objects->points) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.ns = yoloCloud.labels.at(object.label);
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
            text_marker.ns = yoloCloud.labels.at(object.label);
            text_marker.id = 1;
            text_marker.type = 9;
            text_marker.text = yoloCloud.labels.at(object.label);
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

    YoloCloudNode yc_node(n);
    yc_node.model.load("/home/bwilab/Desktop/train/darknet_data/config/model.names", "/home/bwilab/Desktop/train/darknet_data/config/model.cfg", "/home/bwilab/Desktop/train/darknet_data/backup/model.backup");

    image_transport::SubscriberFilter image_sub(yc_node.it,
                                                "/hsrb/head_rgbd_sensor/rgb/image_rect_color",
                                                10,
                                                image_transport::TransportHints("compressed"));
    image_transport::SubscriberFilter depth_sub(yc_node.it,
                                                "/hsrb/head_rgbd_sensor/depth_registered/image",
                                                10);
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), image_sub, depth_sub);
    sync.registerCallback(boost::bind(&YoloCloudNode::data_callback, &yc_node, _1, _2));

    ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && !yc_node.received_first_message) {
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }

    ros::ServiceServer getEntities = n.advertiseService("getEntities", &YoloCloudNode::get_entities, &yc_node);

    ros::spin();
    return 0;
}
