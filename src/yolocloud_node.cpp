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

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#define VISUALIZE true

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image> SyncPolicy;


class YoloCloudNode {
private:
    YoloCloud yoloCloud;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;
    std::unordered_map<int, int> entity_id_to_point;
    octomap::OcTree octree;
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
          octree(0.01),
          ltmc(knowledge_rep::get_default_ltmc()) {

#ifdef VISUALIZE
        viz_yolo_pub = it.advertise("yolocloud/detections", 1);
        viz_pub = node.advertise<visualization_msgs::MarkerArray>("yolocloud/markers", 1, true);
        cloud_pub = node.advertise<octomap_msgs::Octomap>("yolocloud/cloud", 1);
        bbox_pub = node.advertise<visualization_msgs::MarkerArray>("yolocloud/box", 1, true);
        //cloudPCLPub = node.advertise<pcl::PointCloud<pcl::PointXYZ>>("yolocloud/cloudpcl", 1);
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
        std::vector<Detection> dets = model.detect(&cv_ptr->image);

        // Quit if there are no YOLO detections
        if (dets.empty()) {
            return;
        }

        // Parts of the depth image that have YOLO objects
        // This will be useful for constructing a region-of-interest Point Cloud
        cv::Mat depthMasked = cv::Mat::zeros(depth_image->height, depth_image->width, CV_16UC1);

        std::vector<pcl::PointXYZL> detectionPositions;
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

            std::pair<int, pcl::PointXYZL> ret = yoloCloud.addObject(bbox, cv_ptr->image, depthI, camToMap);
            detectionPositions.push_back(ret.second);

            // If object was added to yolocloud, add to knowledge base
            int idx = ret.first;
            if (idx >= 0) {
                add_to_ltmc(idx);
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
                                2);  // Max range of 2. This isn't meters, I don't know wtf this is.

        // Bounding box
        visualization_msgs::MarkerArray boxes;
        for (int i = 0; i < dets.size(); i++) {
            pcl::PointXYZL point = detectionPositions[i];
            visualization_msgs::Marker box = extract_bounding_box(point, dets[i].bbox, camToMap, ir_intrinsics);
            box.id = i;
            boxes.markers.push_back(box);
        }
        bbox_pub.publish(boxes);

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


    visualization_msgs::Marker extract_bounding_box(pcl::PointXYZL &point, cv::Rect &yolobbox, Eigen::Affine3f &camToMap, Eigen::Matrix3f &ir_intrinsics) {
        // This gets the min and max of the rough bounding box to look for object points
        // The min and max is cutoff by the true min and max of our Octomap
        double min_bb_x;
        double min_bb_y;
        double min_bb_z;
        double max_bb_x;
        double max_bb_y;
        double max_bb_z;
        octree.getMetricMin(min_bb_x, min_bb_y, min_bb_z);
        octree.getMetricMax(max_bb_x, max_bb_y, max_bb_z);
        min_bb_x = std::max(min_bb_x, point.x - 0.2);
        min_bb_y = std::max(min_bb_y, point.y - 0.2);
        min_bb_z = std::max(min_bb_z, point.z - 0.3);
        max_bb_x = std::min(max_bb_x, point.x + 0.2);
        max_bb_y = std::min(max_bb_y, point.y + 0.2);
        max_bb_z = std::min(max_bb_z, point.z + 0.3);
        octomap::point3d min(min_bb_x, min_bb_y, min_bb_z);
        octomap::point3d max(max_bb_x, max_bb_y, max_bb_z);

        // Octomap stores free space, so just focus on occupied cells within our rough bounding box
        std::vector<Eigen::Vector3f> pts;
        float minZ = std::numeric_limits<float>::max();
        for (octomap::OcTree::leaf_bbx_iterator iter = octree.begin_leafs_bbx(min, max),
                 end=octree.end_leafs_bbx(); iter != end; ++iter) {
            if (iter->getOccupancy() >= octree.getOccupancyThres()) {
                pts.emplace_back(iter.getCoordinate().x(), iter.getCoordinate().y(), iter.getCoordinate().z());
                minZ = std::min(minZ, iter.getCoordinate().z());
            }
        }

        // Filter out z's that are too low (e.g. if part of the table is captured)
        std::vector<Eigen::Vector3f> ptsZFiltered;
        for (const auto &pt : pts) {
            if (pt(2) > minZ + 0.045) {
                ptsZFiltered.push_back(pt);
            }
        }
        pts = ptsZFiltered;

        // Prepare batch of points to check if in YOLO bounding box
        Eigen::Matrix<float, 3, Eigen::Dynamic> candidatePoints(3, pts.size());
        for (int i = 0; i < pts.size(); ++i) {
            candidatePoints.col(i) = pts[i];
        }

        // Project batch
        Eigen::Affine3f mapToCam = camToMap.inverse();
        Eigen::Matrix<float, 3, Eigen::Dynamic> imagePoints = ir_intrinsics * mapToCam * candidatePoints;
        imagePoints.array().rowwise() /= imagePoints.row(2).array();

        // Construct PCL Point Cloud with subset of points that project into YOLO bounding box
        pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (int i = 0; i < pts.size(); ++i) {
            float x = imagePoints(0, i);
            float y = imagePoints(1, i);

            int x_b = yolobbox.x;
            int y_b = yolobbox.y;

            // Within YOLO Box
            if (x_b <= x && x < x_b + yolobbox.width && y_b <= y && y < y_b + yolobbox.height) {
                objectCloud->points.emplace_back(pts[i](0), pts[i](1), pts[i](2));
            }
        }

        if (objectCloud->points.empty()) {
            return visualization_msgs::Marker();
        }

        // Use Euclidean clustering to filter out noise like objects that are behind our object
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(objectCloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02); // 2cm
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(objectCloud);
        ec.extract(cluster_indices);

        if (cluster_indices.empty()) {
            return visualization_msgs::Marker();
        }

        // TODO: Closest cluster might be better than max...
        auto max_cluster = std::max_element(cluster_indices.begin(),
                                            cluster_indices.end(),
                                            [](const pcl::PointIndices &a, const pcl::PointIndices &b) {
            return a.indices.size() < b.indices.size();
        });

        pcl::PointCloud<pcl::PointXYZ>::Ptr clusteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = max_cluster->indices.begin(); pit != max_cluster->indices.end (); ++pit) {
            pcl::PointXYZ pt = objectCloud->points[*pit];
            clusteredCloud->points.push_back(pt);
        }

        // The following gets and computes the oriented bounding box

        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud(clusteredCloud);
        feature_extractor.compute();

        pcl::PointXYZ min_point_OBB;
        pcl::PointXYZ max_point_OBB;
        pcl::PointXYZ position_OBB;
        Eigen::Matrix3f rotation_matrix_OBB;
        Eigen::Vector3f major_vector, middle_vector, minor_vector;

        feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);
        feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);

        // The orientation returned from getOBB is not what we want
        // There are no constraints, and it rearranges the axes in an arbitrary way
        // Instead, we will compute the rotation ourselves
        // from the eigenvectors of the (covariance of) the object's point cloud

        Eigen::Vector3f eigenvectors[3] = { major_vector, middle_vector, minor_vector };

        // Find the closest eigenvector to each of our principal axes
        // Since we're in a Hilbert space, the inner product tells us this
        Eigen::Vector3f x = Eigen::Vector3f::UnitX();
        Eigen::Vector3f y = Eigen::Vector3f::UnitY();
        Eigen::Vector3f z = Eigen::Vector3f::UnitZ();
        float cos_with_x[3] = {
                std::abs(x.dot(major_vector)),
                std::abs(x.dot(middle_vector)),
                std::abs(x.dot(minor_vector))
        };
        float cos_with_y[3] = {
                std::abs(y.dot(major_vector)),
                std::abs(y.dot(middle_vector)),
                std::abs(y.dot(minor_vector))
        };
        float cos_with_z[3] = {
                std::abs(z.dot(major_vector)),
                std::abs(z.dot(middle_vector)),
                std::abs(z.dot(minor_vector))
        };
        auto x_idx = std::max_element(cos_with_x, cos_with_x+3) - cos_with_x;
        auto y_idx = std::max_element(cos_with_y, cos_with_y+3) - cos_with_y;
        auto z_idx = std::max_element(cos_with_z, cos_with_z+3) - cos_with_z;

        // I now want to rotate our x axis to a "grounded" version
        // of the eigenvector closest to the x axis
        // By grounded I mean projected onto the ground
        // In other words, we're only concerned with a change in yaw
        Eigen::Vector3f grounded_axis(eigenvectors[x_idx]);
        grounded_axis(2) = 0.;
        grounded_axis.normalize();
        Eigen::Quaternionf rotation;
        rotation = Eigen::AngleAxisf(acos(x.dot(grounded_axis)), x.cross(grounded_axis).normalized());

        // getOBB returns bounds for the box, where the first component has the major
        // second the middle, and third the minor axis bounds
        // We want to rearrange this such that the first component is x, second y, and third z
        Eigen::Vector3f orig_min_point_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
        Eigen::Vector3f orig_max_point_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
        Eigen::Vector3f min_point(orig_min_point_OBB(x_idx), orig_min_point_OBB(y_idx), orig_min_point_OBB(z_idx));
        Eigen::Vector3f max_point(orig_max_point_OBB(x_idx), orig_max_point_OBB(y_idx), orig_max_point_OBB(z_idx));

        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.id = 1;
        marker.type = 1;
        marker.action = 0;
        marker.pose.position.x = position_OBB.x;
        marker.pose.position.y = position_OBB.y;
        marker.pose.position.z = position_OBB.z;
        marker.pose.orientation.x = rotation.x();
        marker.pose.orientation.y = rotation.y();
        marker.pose.orientation.z = rotation.z();
        marker.pose.orientation.w = rotation.w();
        marker.scale.x = max_point(0) - min_point(0);
        marker.scale.y = max_point(1) - min_point(1);
        marker.scale.z = max_point(2) - min_point(2);
        marker.color.r = 0.;
        marker.color.b = 0.;
        marker.color.g = 1.;
        marker.color.a = 1.;
        marker.lifetime = ros::Duration(0);

        return marker;
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
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }

    ros::ServiceServer getEntities = n.advertiseService("getEntities", &YoloCloudNode::get_entities, &yc_node);

    ros::spin();
    return 0;
}
