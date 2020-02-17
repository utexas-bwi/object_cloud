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
#include <nav_msgs/Odometry.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <knowledge_representation/LongTermMemoryConduit.h>
#include <knowledge_representation/convenience.h>
#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <knowledge_representation/LTMCInstance.h>
#include <villa_object_cloud/YoloCloudNode.h>
#include <villa_object_cloud/DetectedObject.h>
#include <villa_object_cloud/GetEntities.h>
#include <villa_object_cloud/GetBoundingBoxes.h>
#include <villa_object_cloud/GetObjects.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
        sensor_msgs::Image,
        nav_msgs::Odometry> SyncPolicy;

int main(int argc, char **argv) {
    ROS_INFO("Initializing yolocloud node...");
    ros::init(argc, argv, "yolocloud_node");
    ros::NodeHandle n("~");

    // Network config/weights
    std::string labels_file;
    std::string network_cfg;
    std::string weights_file;
    n.getParam("yolo_labels", labels_file);
    n.getParam("yolo_cfg", network_cfg);
    n.getParam("yolo_weights", weights_file);

    YoloCloudNode yc_node(n);
    yc_node.model.load(labels_file, network_cfg, weights_file);

    // RGBD Subscriber
    image_transport::SubscriberFilter image_sub(yc_node.it,
                                                "/hsrb/head_rgbd_sensor/rgb/image_rect_color",
                                                30,
                                                image_transport::TransportHints("compressed"));
    image_transport::SubscriberFilter depth_sub(yc_node.it,
                                                "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw",
                                                30);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/hsrb/odom", 30);
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(30), image_sub, depth_sub, odom_sub);
    sync.registerCallback(boost::bind(&YoloCloudNode::data_callback, &yc_node, _1, _2, _3));

    // Head Subscriber
    image_transport::Subscriber hand_sub = yc_node.it.subscribe("/hsrb/hand_camera/image_raw",
                                                                30,
                                                                boost::bind(&YoloCloudNode::hand_camera_callback,
                                                                            &yc_node, _1));

    ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && !yc_node.received_first_message) {
        ROS_WARN_THROTTLE(2, "Waiting for image and depth messages...");
        ros::spinOnce();
    }
  yc_node.advertise_services();

    // This way we can have one callback thread and one service thread
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    return 0;
}
