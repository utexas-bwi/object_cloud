#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <object_cloud/ColorBlobCloudNode.h>
#include <sensor_msgs/CameraInfo.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry>
    SyncPolicy;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "color_blob_cloud_node");
  ros::NodeHandle n("~");

  ROS_INFO("Waiting for camera info...");
  // We need to know the camera's projection matrix. If you're doing things the ROS way,
  // you've calibrated the camera and this information will appear in a camera info topic
  auto cam_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/depth/camera_info", n);
  // Message is rom major but Eigen is column major
  Eigen::Matrix3f camera_intrinsics = Eigen::Matrix3d(cam_info->K.data()).transpose().cast<float>();

  ColorBlobCloudNode cbc_node(n, camera_intrinsics);

  image_transport::SubscriberFilter image_sub(cbc_node.it, "/rgb/image", 5);
  image_transport::SubscriberFilter depth_sub(cbc_node.it, "/depth/image", 5);
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/odom", 5);
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(5), image_sub, depth_sub, odom_sub);
  sync.registerCallback(boost::bind(&ColorBlobCloudNode::dataCallback, &cbc_node, _1, _2, _3));

  ROS_INFO("Started. Waiting for sensor data.");
  while (ros::ok() && !cbc_node.received_first_message)
  {
    ROS_WARN_THROTTLE(2, "Waiting for image and depth messages...");
    ros::spinOnce();
  }
  cbc_node.advertiseServices();

  // This way we can have one callback thread and one service thread
  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();
  return 0;
}
