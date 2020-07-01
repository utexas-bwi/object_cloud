#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <object_cloud/ColorBlobCloudNode.h>

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry>
    SyncPolicy;

int main(int argc, char **argv) {
  ROS_INFO("Initializing ground_truth_cloud node...");
  ros::init(argc, argv, "color_blob_cloud_node");
  ros::NodeHandle n("~");

  ColorBlobCloudNode cbc_node(n);

  // RGBD Subscriber
  image_transport::SubscriberFilter image_sub(
      cbc_node.it, "/rgb/image", 30);
  image_transport::SubscriberFilter depth_sub(
      cbc_node.it, "/depth/image",
      30);
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/odom", 30);
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(30), image_sub,
                                                 depth_sub, odom_sub);
  sync.registerCallback(boost::bind(&ColorBlobCloudNode::data_callback,
                                    &cbc_node, _1, _2, _3));

  ROS_INFO("Started. Waiting for inputs.");
  while (ros::ok() && !cbc_node.received_first_message) {
    ROS_WARN_THROTTLE(2, "Waiting for image and depth messages...");
    ros::spinOnce();
  }
  cbc_node.advertise_services();

  // This way we can have one callback thread and one service thread
  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();
  return 0;
}
