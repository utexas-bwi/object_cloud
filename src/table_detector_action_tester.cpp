#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include "../../villa_object_label/include/ObjectLabeler.h"
#include "../../villa_object_label/src/objectlabel.cpp"
#include "std_msgs/String.h"
//#include "villa_object_label/ObjectLabeler.h"


#include <villa_surface_detectors/PerceiveTableAction.h>

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

ros::NodeHandle *n = nullptr;
ObjectLabeler *object_labeler = nullptr;
ros::Publisher object_labels;

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
	tf::TransformListener listener;

	tf::StampedTransform transf;
	ros::Time now = ros::Time::now();
	// listener.waitForTransform("head_rgbd_sensor_rgb_frame", "map", now, ros::Duration(3.0));
	listener.waitForTransform("map", (*input).header.frame_id, now, ros::Duration(5.0));
	listener.lookupTransform("map", (*input).header.frame_id, now, transf);
	sensor_msgs::PointCloud2 input_transformed;
	pcl_ros::transformPointCloud("map", transf, *input, input_transformed);
	// pcl_ros::transformPointCloud("map", *input, input_transformed, listener);

	// Store Cloud
	//cloud_mutex.lock ();

	cloud_in = input_transformed;
	ROS_INFO("Received cloud.");
	//cloud_mutex.unlock ();
	ROS_INFO("Pointcloud is ready.");

	// create the action client
	// true causes the client to spin its own thread
	ROS_INFO("Starting action client");
	actionlib::SimpleActionClient<villa_surface_detectors::PerceiveTableAction> ac("perceive_table", true);

	// Prepare action request
  // wait for the action server to start
	ROS_INFO("Waiting for server to connect");
  ac.waitForServer(); //will wait for infinite time
  ROS_INFO("Server connected: %d", ac.isServerConnected());

	//pressEnter("Press Enter to continue");

  ROS_INFO("Action server started, sending goal.");
  // send a goal to the action
  villa_surface_detectors::PerceiveTableGoal goal;
  goal.cloud_input = cloud_in;
  ac.sendGoal(goal);

  //wait for the action to return
  bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

  if (finished_before_timeout)
  {
    actionlib::SimpleClientGoalState state = ac.getState();

	std::vector<visualization_msgs::Marker> boxes = ac.getResult()->object_bounding_boxes;

	std::vector<DetectedObject> objects;

	for (auto box : boxes) {
		DetectedObject detectedobject(box);
		objects.push_back(detectedobject);
	}

	std::map<std::string, std::string> labels = object_labeler->label_objects(objects, listener);
	string output = "";
	for (auto& kv : labels){
		output += kv.first + " : " + kv.second + "\n";
	}
	std_msgs::String msg;
	msg.data = output;
	object_labels.publish(msg);
	ROS_INFO("Action finished: %s",state.toString().c_str());


  }
  else
    ROS_INFO("Action did not finish before the time out.");
}

int main (int argc, char **argv)
{
	ros::init(argc, argv, "test_action");
	ROS_INFO("Initializing action_tester...");

	n = new ros::NodeHandle();

    std::string param_topic = "/hsrb/head_rgbd_sensor/depth_registered/rectified_points";
	// std::string param_topic = "/octomap_point_cloud_centers";
	
    ros::Subscriber sub = n->subscribe (param_topic, 1, cloud_cb);
	octomap_pub = n->advertise<sensor_msgs::PointCloud2>("octomap_cloud", 1, true);
	octomap_client = n->serviceClient<villa_octomap_server::GetPointCloud>("octomap_cloud");
	object_labeler = new ObjectLabeler(*n, "", "xtion", true);
	object_labels = n->advertise<std_msgs::String>("object_labels", 1000);

	ROS_INFO("Started. Waiting for inputs.");
    ros::spin();

  return 0;
}
