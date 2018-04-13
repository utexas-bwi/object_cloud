#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

#include <villa_surface_detectors/PerceiveShelfAction.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <villa_octomap_server/GetPointCloud.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

sensor_msgs::PointCloud2 cloud_in;

ros::ServiceClient octomap_client;
ros::Publisher octomap_pub;

void pressEnter(std::string message){
	std::cout << message;
	while (true){
		char c = std::cin.get();
		if (c == '\n')
			break;
		else if (c == 'q'){
			ros::shutdown();
			exit(1);
		}
		else {
			std::cout <<  message;
		}
	}
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input){
	// Store Cloud
	//cloud_mutex.lock ();
	ROS_INFO("Ready to request OctoMap.");

	// Prepare service request
	villa_octomap_server::GetPointCloud srv;

	pressEnter("Press enter");
	// Perform Service Call
	if (octomap_client.call(srv)){
		ROS_INFO("Call was successful!\n");
	}else{
		ROS_ERROR("Failed to call service octomap_cloud");
		ROS_INFO("Attempting service call again when cloud becomes available");
	}

	octomap_pub.publish(srv.response.cloud);

	cloud_in = (srv.response.cloud);

	ROS_INFO("Received cloud.");
	//cloud_mutex.unlock ();
	ROS_INFO("Pointcloud is ready.");

	// create the action client
	// true causes the client to spin its own thread
	ROS_INFO("Starting action client");
	actionlib::SimpleActionClient<villa_surface_detectors::PerceiveShelfAction> ac("perceive_shelf", true);

	// Prepare action request
  // wait for the action server to start
	ROS_INFO("Waiting for server to connect");
  ac.waitForServer(); //will wait for infinite time
  ROS_INFO("Server connected: %d", ac.isServerConnected());

	//pressEnter("Press Enter to continue");

  ROS_INFO("Action server started, sending goal.");
  // send a goal to the action
  villa_surface_detectors::PerceiveShelfGoal goal;
  goal.cloud_input = cloud_in;
  ac.sendGoal(goal);

  //wait for the action to return
  bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

  if (finished_before_timeout)
  {
    actionlib::SimpleClientGoalState state = ac.getState();
    ROS_INFO("Action finished: %s",state.toString().c_str());
  }
  else
    ROS_INFO("Action did not finish before the time out.");
}

int main (int argc, char **argv)
{
	ROS_INFO("Initializing action_tester...");
  ros::init(argc, argv, "test_action");
	ros::NodeHandle n;

        //std::string param_topic = "/hsrb/head_rgbd_sensor/depth_registered/rectified_points";
	std::string param_topic = "/octomap_point_cloud_centers";

	ros::Subscriber sub = n.subscribe (param_topic, 1, cloud_cb);
	octomap_pub = n.advertise<sensor_msgs::PointCloud2>("octomap_cloud", 1, true);
	octomap_client = n.serviceClient<villa_octomap_server::GetPointCloud>("octomap_cloud");

	ROS_INFO("Started. Waiting for inputs.");
  ros::spin();

  return 0;
}
