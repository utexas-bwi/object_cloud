#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <object_cloud/GroundTruthObjectCloudNode.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry>
    SyncPolicy;

int main(int argc, char** argv)
{
  ROS_INFO("Initializing ground_truth_cloud node...");
  ros::init(argc, argv, "ground_truth_cloud_node");
  ros::NodeHandle n("~");

  Eigen::Matrix3f camera_intrinsics;
  camera_intrinsics << 535.2900990271, 0, 320.0, 0, 535.2900990271, 240.0, 0, 0, 1;
  GroundTruthObjectCloudNode gtc_node(n, camera_intrinsics);

  // RGBD Subscriber
  image_transport::SubscriberFilter image_sub(gtc_node.it, "/hsrb/head_rgbd_sensor/rgb/image_rect_color", 30,
                                              image_transport::TransportHints("compressed"));
  image_transport::SubscriberFilter depth_sub(gtc_node.it, "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw",
                                              30);
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/hsrb/odom", 30);
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(30), image_sub, depth_sub, odom_sub);
  sync.registerCallback(boost::bind(&GroundTruthObjectCloudNode::dataCallback, &gtc_node, _1, _2, _3));

  // Head Subscriber
  image_transport::Subscriber hand_sub = gtc_node.it.subscribe(
      "/hsrb/hand_camera/image_raw", 30, boost::bind(&GroundTruthObjectCloudNode::handCameraCallback, &gtc_node, _1));

  ROS_INFO("Started. Waiting for inputs.");
  while (ros::ok() && !gtc_node.received_first_message)
  {
    ROS_WARN_THROTTLE(2, "Waiting for image and depth messages...");
    ros::spinOnce();
  }
  gtc_node.advertiseServices();

  // This way we can have one callback thread and one service thread
  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();
  return 0;
}
