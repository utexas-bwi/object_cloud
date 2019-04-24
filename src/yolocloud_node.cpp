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
#include <sensor_msgs/PointCloud2.h>
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
#include <villa_yolocloud/PointCloudConstructor.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv/highgui.h>
#include <chrono>

#define VISUALIZE true

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image> SyncPolicy;


class YoloCloudNode {
private:
    YoloCloud yoloCloud;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;
    std::unordered_map<int, octomap::point3d> entity_id_to_points;
    std::unordered_map<octomap::OcTreeKey, visualization_msgs::Marker, octomap::OcTreeKey::KeyHash> bounding_boxes;
    knowledge_rep::LongTermMemoryConduit ltmc; // Knowledge base


#ifdef VISUALIZE
    image_transport::Publisher viz_yolo_pub;
    ros::Publisher viz_pub;
    ros::Publisher bbox_pub;
    ros::Publisher cloud_pub;
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
        cloud_pub = node.advertise<octomap_msgs::Octomap>("yolocloud/cloud", 1);
        bbox_pub = node.advertise<visualization_msgs::MarkerArray>("yolocloud/box", 1);
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
        octomap::OcTree &octree = yoloCloud.octree;
        received_first_message = true;
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }


        // Beware! head_rgbd_sensor_rgb_frame is different from head_rgbd_sensor_link
        geometry_msgs::TransformStamped transform;
        try {
            transform = tfBuffer.lookupTransform("map", rgb_image->header.frame_id, rgb_image->header.stamp, ros::Duration(0.01));
        } catch (tf2::TransformException &ex){
            ROS_ERROR("%s", ex.what());
            return;
        }

        Eigen::Affine3f camToMap = tf2::transformToEigen(transform).cast<float>();

        cv::Mat depthI(depth_image->height, depth_image->width, CV_16UC1);
        memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Octomap gets confused by partial views of objects
        // We could check if the bounding box touches the border of the image to prevent this
        std::vector<Detection> dets = model.detect(&cv_ptr->image);

        // Quit if there are no YOLO detections
        if (dets.empty()) {
            return;
        }

        // Parts of the depth image that have YOLO objects
        // This will be useful for constructing a region-of-interest Point Cloud
        cv::Mat depthMasked = cv::Mat::zeros(depth_image->height, depth_image->width, CV_16UC1);

        std::vector<octomap::point3d> detectionPositions;
        for (const auto &detection : dets) {
            ImageBoundingBox bbox;
            bbox.x = detection.bbox.x;
            bbox.y = detection.bbox.y;
            bbox.width = detection.bbox.width;
            bbox.height = detection.bbox.height;
            bbox.label = detection.label;

            // 10 pixel buffer
            int min_x = std::max(0, detection.bbox.x - 10);
            int min_y = std::max(0, detection.bbox.y - 10);
            int max_x = std::min(cv_ptr->image.cols-1, detection.bbox.x + detection.bbox.width + 20);
            int max_y = std::min(cv_ptr->image.rows-1, detection.bbox.y + detection.bbox.height + 20);
            cv::Rect region(min_x, min_y, max_x - min_x, max_y - min_y);
            depthI(region).copyTo(depthMasked(region));

            std::pair<bool, YoloCloudObject> ret = yoloCloud.addObject(bbox, cv_ptr->image, depthI, camToMap);
            detectionPositions.push_back(ret.second.position);

            // If object was added to yolocloud, add to knowledge base
            bool newObj = ret.first;
            if (newObj) {
                add_to_ltmc(ret.second);
            }
        }

        // Use depthMasked to construct a ROI Point Cloud for use with Octomap
        // Without this, Octomap takes forever
        Eigen::Matrix3f ir_intrinsics;
        ir_intrinsics << 535.2900990271, 0, 320.0, 0, 535.2900990271, 240.0, 0, 0, 1;
        octomap::Pointcloud cloud = PointCloudConstructor::construct(ir_intrinsics, depthMasked, camToMap);

        // Insert ROI PointCloud into Octree
        octree.insertPointCloud(cloud,
                                octomap::point3d(0, 0, 0),
                                2,       // Max range of 2. This isn't meters, I don't know wtf this is.
                                false,   // We don't want lazy updates
                                true);   // Discretize speeds it up by approximating

        // Bounding box
        for (int i = 0; i < dets.size(); i++) {
            octomap::point3d point = detectionPositions[i];
            visualization_msgs::Marker box = yoloCloud.extractBoundingBox(point, dets[i].bbox, camToMap, ir_intrinsics);

            // Invalid box
            if (box.header.frame_id.empty()) {
                continue;
            }

            auto mit = bounding_boxes.find(yoloCloud.octree.coordToKey(point));
            if (mit != bounding_boxes.end()) {
                mit->second = box;
            } else {
                bounding_boxes.insert({yoloCloud.octree.coordToKey(point), box});
            }

        }

        // Record end time
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";

#ifdef VISUALIZE
        // Publish Octree
        octomap_msgs::Octomap cloudMsg;
        cloudMsg.header.frame_id = "map";
        octomap_msgs::binaryMapToMsg(octree, cloudMsg);
        cloud_pub.publish(cloudMsg);

        // Publish bounding boxes
        visualization_msgs::MarkerArray boxes;
        int id = 0;
        for (const auto &e : bounding_boxes) {
            visualization_msgs::Marker box = e.second;
            box.id = id;
            boxes.markers.push_back(box);
            id++;
        }
        bbox_pub.publish(boxes);

        // Publish YOLO
        for (const auto &d : dets) {
            cv::rectangle(cv_ptr->image, d.bbox, cv::Scalar(100, 100, 0), 3);
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
        viz_yolo_pub.publish(msg);

        // Publish cloud
        visualize();
#endif
    }


    void add_to_ltmc(const YoloCloudObject &object) {
        auto concept = ltmc.get_concept(object.label);
        auto entity = concept.create_instance();
        auto sensed = ltmc.get_concept("sensed");
        entity.make_instance_of(sensed);
        entity_id_to_points.insert({entity.entity_id, object.position});
    }


    bool get_entities(villa_yolocloud::GetEntities::Request &req, villa_yolocloud::GetEntities::Response &res) {
        std::vector<geometry_msgs::Point> map_locations;
        for (int eid : req.entity_ids) {
            geometry_msgs::Point p;
            // Requested an ID that's not in the cloud. Return NANs
            if (entity_id_to_points.count(eid) == 0) {
                knowledge_rep::Entity entity = {eid, ltmc};
                entity.remove_attribute("sensed");
                p.x = NAN;
                p.y = NAN;
                p.z = NAN;
            } else {
                octomap::point3d point = entity_id_to_points.at(eid);
                p.x = point.x();
                p.y = point.y();
                p.z = point.z();
            }

            map_locations.push_back(p);
        }

        res.map_locations = map_locations;

        return true;
    }


    void visualize() {
        visualization_msgs::MarkerArray yolo_marker_array;
        for (const auto &object : yoloCloud.getAllObjects()) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.ns = object.label;
            marker.id = 0;
            marker.type = 1;
            marker.action = 0;
            marker.pose.position.x = object.position.x();
            marker.pose.position.y = object.position.y();
            marker.pose.position.z = object.position.z();
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
            text_marker.ns = object.label;
            text_marker.id = 1;
            text_marker.type = 9;
            text_marker.text = object.label;
            text_marker.action = 0;
            text_marker.pose.position.x = object.position.x();
            text_marker.pose.position.y = object.position.y();
            text_marker.pose.position.z = object.position.z() + 0.10;
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

    image_transport::SubscriberFilter image_sub(yc_node.it,
                                                "/hsrb/head_rgbd_sensor/rgb/image_rect_color",
                                                10,
                                                image_transport::TransportHints("compressed"));
    image_transport::SubscriberFilter depth_sub(yc_node.it,
                                                "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw",
                                                10);
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), image_sub, depth_sub);
    sync.registerCallback(boost::bind(&YoloCloudNode::data_callback, &yc_node, _1, _2));

    ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && !yc_node.received_first_message) {
        ROS_WARN_THROTTLE(2, "Waiting for image and depth messages...");
        ros::spinOnce();
    }

    ros::ServiceServer getEntities = n.advertiseService("getEntities", &YoloCloudNode::get_entities, &yc_node);

    ros::spin();
    return 0;
}
