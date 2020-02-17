#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <knowledge_representation/convenience.h>
#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <knowledge_representation/LTMCInstance.h>
#include <villa_object_cloud/PointCloudConstructor.h>
#include <villa_object_cloud/PointCloudUtils.h>
#include <villa_object_cloud/YoloCloudNode.h>
#include <villa_object_cloud/BoundingBox2DList.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv/highgui.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

YoloCloudNode::YoloCloudNode(ros::NodeHandle node)
    : ObjectCloudNode(node) {

    // Set up publishers per camera
    detections_pubs.insert({"head_rgbd", node.advertise<villa_object_cloud::BoundingBox2DList>("/object_cloud/detections/head_rgbd", 10)});
    detections_pubs.insert({"hand", node.advertise<villa_object_cloud::BoundingBox2DList>("/object_cloud/detections/hand", 10)});

}

void YoloCloudNode::data_callback(const sensor_msgs::Image::ConstPtr &rgb_image,
                                  const sensor_msgs::Image::ConstPtr &depth_image,
                                  const nav_msgs::Odometry::ConstPtr &odom) {
    std::lock_guard<std::mutex> global_lock(global_mutex);

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    received_first_message = true;
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }


    // Beware! head_rgbd_sensor_rgb_frame is different from head_rgbd_sensor_link
    geometry_msgs::TransformStamped camToMapTransform;
    geometry_msgs::TransformStamped baseToMapTransform;
    try {
      camToMapTransform = tf_buffer.lookupTransform("map", rgb_image->header.frame_id, rgb_image->header.stamp,
                                                    ros::Duration(0.02));
      baseToMapTransform = tf_buffer.lookupTransform("map", "base_link", rgb_image->header.stamp,
                                                     ros::Duration(0.02));
    } catch (tf2::TransformException &ex) {
        ROS_ERROR("%s", ex.what());
        return;
    }

    Eigen::Affine3f camToMap = tf2::transformToEigen(camToMapTransform).cast<float>();
    Eigen::Affine3f baseToMap = tf2::transformToEigen(baseToMapTransform).cast<float>();

    cv::Mat depthI(depth_image->height, depth_image->width, CV_16UC1);
    memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());

    // Process point cloud requests
    Eigen::Matrix3f ir_intrinsics;
    ir_intrinsics << 535.2900990271, 0, 320.0, 0, 535.2900990271, 240.0, 0, 0, 1;
    float inf = std::numeric_limits<float>::infinity();
  while (!point_cloud_requests.empty()) {
    std::shared_ptr<PointCloudRequest> req = point_cloud_requests.pop();
        std::cout << "PROCESSING" << std::endl;
        {
            std::lock_guard<std::mutex> lock(req->mutex);

          octomap::Pointcloud planecloud = PointCloudConstructor::construct(ir_intrinsics, depthI, req->cam_to_target,
                                                                            inf,
                                                                            req->xbounds,
                                                                            req->ybounds,
                                                                            req->zbounds);
            pcl::PointCloud<pcl::PointXYZ>::Ptr req_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            req_cloud->points.reserve(planecloud.size());
            for (const auto &p : planecloud) {
                req_cloud->points.emplace_back(p.x(), p.y(), p.z());
            }
            req->result = req_cloud;
        }
        std::cout << "PROCESSED" << std::endl;

        req->cond_var.notify_one();
        std::cout << "NOTIFIED" << std::endl;
    }

    std::vector<Detection> dets;

    {
        std::lock_guard<std::mutex> yolo_lock(yolo_mutex);
        dets = model.detect(&cv_ptr->image);
    }

    // Quit if there are no YOLO detections
    if (dets.empty()) {
        return;
    }

    // Publish yolo detections
    villa_object_cloud::BoundingBox2DList detections_msg;
    detections_msg.bbox_list.resize(dets.size());
    for (int i = 0; i < dets.size(); ++i) {
        detections_msg.bbox_list[i].x = dets[i].bbox.x;
        detections_msg.bbox_list[i].y = dets[i].bbox.y;
        detections_msg.bbox_list[i].width = dets[i].bbox.width;
        detections_msg.bbox_list[i].height = dets[i].bbox.height;
        detections_msg.bbox_list[i].label = dets[i].label;
    }
    detections_pubs["head_rgbd"].publish(detections_msg);

    // Parts of the depth image that have YOLO objects
    // This will be useful for constructing a region-of-interest Point Cloud
    cv::Mat depthMasked = cv::Mat::zeros(depth_image->height, depth_image->width, CV_16UC1);

  std::vector<std::pair<Detection, Object>> detectionPositions;
    for (const auto &detection : dets) {
        ImageBoundingBox bbox;
        bbox.x = detection.bbox.x;
        bbox.y = detection.bbox.y;
        bbox.width = detection.bbox.width;
        bbox.height = detection.bbox.height;
        bbox.label = detection.label;

        // If the bounding box is at the edge of the image, ignore it
        if (bbox.x == 0 || bbox.x + bbox.width >= cv_ptr->image.cols
            || bbox.y == 0 || bbox.y + bbox.height >= cv_ptr->image.rows) {
            continue;
        }

        // 10 pixel buffer
        int min_x = std::max(0, detection.bbox.x - 10);
        int min_y = std::max(0, detection.bbox.y - 10);
        int max_x = std::min(cv_ptr->image.cols - 1, detection.bbox.x + detection.bbox.width + 20);
        int max_y = std::min(cv_ptr->image.rows - 1, detection.bbox.y + detection.bbox.height + 20);
        cv::Rect region(min_x, min_y, max_x - min_x, max_y - min_y);
        depthI(region).copyTo(depthMasked(region));

      std::pair<bool, Object> ret = object_cloud.add_object(bbox, cv_ptr->image, depthI, camToMap);
        if (!ret.second.invalid()) {
            detectionPositions.emplace_back(detection, ret.second);
        }

        // If object was added to object_cloud, add to knowledge base
        bool newObj = ret.first;
        if (newObj) {
            std::cout << "New Object " << ret.second.position << std::endl;
            add_to_ltmc(ret.second);
        }
    }

    // If the robot is moving then don't update Octomap
    if (Eigen::Vector3f(odom->twist.twist.linear.x,
                        odom->twist.twist.linear.y,
                        odom->twist.twist.linear.z).norm() > 0.05 ||
        Eigen::Vector3f(odom->twist.twist.angular.x,
                        odom->twist.twist.angular.y,
                        odom->twist.twist.angular.z).norm() > 0.05) {
        return;
    }

    // Use depthMasked to construct a ROI Point Cloud for use with Octomap
    // Without this, Octomap takes forever
    Eigen::Vector2f nobounds(-inf, inf);
    octomap::Pointcloud cloud = PointCloudConstructor::construct(ir_intrinsics, depthMasked, camToMap,
                                                                 3.,
                                                                 nobounds,
                                                                 nobounds,
                                                                 Eigen::Vector2f(0., inf));

    // Insert ROI PointCloud into Octree
    Eigen::Vector3f origin = camToMap * Eigen::Vector3f::Zero();
    octree.insertPointCloud(cloud,
                            octomap::point3d(origin(0), origin(1), origin(2)),
                            3,       // Max range of 3. This isn't meters, I don't know wtf this is.
                            false,   // We don't want lazy updates
                            true);   // Discretize speeds it up by approximating

    // Bounding box
    for (const auto &d : detectionPositions) {
        octomap::point3d point(d.second.position(0), d.second.position(1), d.second.position(2));
        cv::Rect bbox = d.first.bbox;
        visualization_msgs::Marker box = PointCloudUtils::extractBoundingBox(octree, point, bbox, camToMap, baseToMap,
                                                                             ir_intrinsics);

        // Invalid box
        if (box.header.frame_id.empty()) {
            continue;
        }

        int key = d.second.id;
        auto mit = bounding_boxes.find(key);
        if (mit != bounding_boxes.end()) {
            bounding_boxes.at(key) = box;
        } else {
            bounding_boxes.insert({key, box});
        }

    }

#if(VISUALIZE)

// This takes time so disable if not debugging...
#if(VISUALIZE_OCTREE)
    // Publish Octree
    octomap_msgs::Octomap cloudMsg;
    cloudMsg.header.frame_id = "map";
    octomap_msgs::binaryMapToMsg(octree, cloudMsg);
    cloud_pub.publish(cloudMsg);
#endif

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
    viz_detections_pub.publish(msg);

    // Publish cloud
    visualize();
#endif

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}


void YoloCloudNode::hand_camera_callback(const sensor_msgs::Image::ConstPtr &rgb_image) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Rotate image 90 degrees since hand cam is already rotated for some reason
    cv::Mat transposed_image;
    cv::Mat rotated_image;
    cv::transpose(cv_ptr->image, transposed_image);
    cv::flip(transposed_image, rotated_image, 0);

    std::vector<Detection> dets;

    {
        std::lock_guard<std::mutex> yolo_lock(yolo_mutex);
        dets = model.detect(&rotated_image);
    }

    // Quit if there are no YOLO detections
    if (dets.empty()) {
        villa_object_cloud::BoundingBox2DList detections_msg;
        detections_pubs["hand"].publish(detections_msg);
        return;
    }

    // Publish yolo detections
    villa_object_cloud::BoundingBox2DList detections_msg;
    detections_msg.bbox_list.resize(dets.size());
    for (int i = 0; i < dets.size(); ++i) {
        // Undo rotation (transpose and flip y)
        // Pick the corner on new top left
        int y1 = dets[i].bbox.x;
        int x1 = dets[i].bbox.y;
        x1 = rotated_image.rows - 1 - x1;
        int y2 = dets[i].bbox.x + dets[i].bbox.width - 1;
        int x2 = dets[i].bbox.y;
        x2 = rotated_image.rows - 1 - x2;
        int y3 = dets[i].bbox.x;
        int x3 = dets[i].bbox.y + dets[i].bbox.height - 1;
        x3 = rotated_image.rows - 1 - x3;
        int y4 = dets[i].bbox.x + dets[i].bbox.width - 1;
        int x4 = dets[i].bbox.y + dets[i].bbox.height - 1;
        x4 = rotated_image.rows - 1 - x4;

        int x = std::min({x1, x2, x3, x4});
        int y = std::min({y1, y2, y3, y4});

        int bbox_width = dets[i].bbox.height;  // Flip width and height since we rotated
        int bbox_height = dets[i].bbox.width;  // Flip width and height since we rotated
        detections_msg.bbox_list[i].x = x;
        detections_msg.bbox_list[i].y = y;
        detections_msg.bbox_list[i].width = bbox_width;
        detections_msg.bbox_list[i].height = bbox_height;
        detections_msg.bbox_list[i].label = dets[i].label;
    }
    detections_pubs["hand"].publish(detections_msg);
}


