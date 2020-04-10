#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <knowledge_representation/LTMCInstance.h>
#include <knowledge_representation/convenience.h>
#include <object_cloud/BoundingBox2DList.h>
#include <object_cloud/ObjectCloudNode.h>
#include <object_cloud/PointCloudConstructor.h>
#include <object_cloud/PointCloudUtils.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <opencv/highgui.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

ObjectCloudNode::ObjectCloudNode(ros::NodeHandle node)
    : node(node), tf_listener(tf_buffer), it(node), octree(0.01),
      ltmc(knowledge_rep::getDefaultLTMC()) {

#ifdef VISUALIZE
  viz_detections_pub = it.advertise("/object_cloud/detections", 1);
  viz_pub = node.advertise<visualization_msgs::MarkerArray>(
      "/object_cloud/markers", 1, true);
  cloud_pub = node.advertise<octomap_msgs::Octomap>("/object_cloud/cloud", 1);
  surfacecloud_pub =
      node.advertise<PointCloudT>("/object_cloud/surfacecloud", 1);
  surface_marker_pub = node.advertise<visualization_msgs::MarkerArray>(
      "/object_cloud/surfacemarker", 1);
  bbox_pub =
      node.advertise<visualization_msgs::MarkerArray>("/object_cloud/box", 1);
#endif

  // NOTE: We assume there's only one ObjectCloud, so this will blow away
  // anything that is sensed
  ltmc.getConcept("sensed").removeInstances();
  ltmc.getConcept("scanned").removeReferences();
}

ObjectCloudNode::~ObjectCloudNode() {
  ltmc.getConcept("sensed").removeInstances();
  ltmc.getConcept("scanned").removeReferences();
}

void ObjectCloudNode::advertise_services() {
  // TODO: Switch to standard service naming convention:
  // http://wiki.ros.org/ROS/Patterns/Conventions#Topics_.2BAC8_services
  clear_octree_server = node.advertiseService(
      "/object_cloud/clear_octree", &ObjectCloudNode::clear_octree, this);
  get_entities_server = node.advertiseService(
      "/object_cloud/get_entities", &ObjectCloudNode::get_entities, this);
  get_bounding_boxes_server =
      node.advertiseService("/object_cloud/get_bounding_boxes",
                            &ObjectCloudNode::get_bounding_boxes, this);
  get_objects_server = node.advertiseService(
      "/object_cloud/get_objects", &ObjectCloudNode::get_objects, this);
  get_surface_server = node.advertiseService(
      "/object_cloud/get_surfaces", &ObjectCloudNode::get_surfaces, this);
  surface_occupancy_server =
      node.advertiseService("/object_cloud/get_surface_occupancy",
                            &ObjectCloudNode::get_surface_occupancy, this);
}

void ObjectCloudNode::add_to_ltmc(const Object &object) {
  auto concept = ltmc.getConcept(object.label);
  auto entity = concept.createInstance();
  auto sensed = ltmc.getConcept("sensed");
  entity.makeInstanceOf(sensed);
  entity_id_to_object.insert({entity.entity_id, object});
}

bool ObjectCloudNode::get_entities(object_cloud::GetEntities::Request &req,
                                   object_cloud::GetEntities::Response &res) {
  std::vector<geometry_msgs::Point> map_locations;
  for (int eid : req.entity_ids) {
    geometry_msgs::Point p;
    // Requested an ID that's not in the cloud. Return NANs
    if (entity_id_to_object.count(eid) == 0) {
      knowledge_rep::Entity entity = {eid, ltmc};
      entity.removeAttribute("sensed");
      p.x = NAN;
      p.y = NAN;
      p.z = NAN;
    } else {
      Eigen::Vector3f point = entity_id_to_object.at(eid).position;
      p.x = point(0);
      p.y = point(1);
      p.z = point(2);
    }

    map_locations.push_back(p);
  }

  res.map_locations = map_locations;

  return true;
}

bool ObjectCloudNode::clear_octree(std_srvs::Empty::Request &req,
                                   std_srvs::Empty::Response &res) {
  std::lock_guard<std::mutex> lock(global_mutex);

  octree.clear();
  bounding_boxes.clear();
  return true;
}

bool ObjectCloudNode::get_objects(object_cloud::GetObjects::Request &req,
                                  object_cloud::GetObjects::Response &res) {
  geometry_msgs::TransformStamped frameToMapTransform;
  try {
    frameToMapTransform = tf_buffer.lookupTransform(
        "map", req.search_box.header.frame_id, req.search_box.header.stamp,
        ros::Duration(0.1));
  } catch (tf2::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return false;
  }
  Eigen::Affine3f frameToMap =
      tf2::transformToEigen(frameToMapTransform).cast<float>();
  Eigen::Vector3f origin(req.search_box.origin.x, req.search_box.origin.y,
                         req.search_box.origin.z);
  Eigen::Vector3f scale(req.search_box.scale.x, req.search_box.scale.y,
                        req.search_box.scale.z);

  std::vector<Object> objects =
      object_cloud.search_box(origin, scale, frameToMap);

  for (const auto &object : objects) {
    object_cloud::DetectedObject obj;
    obj.header.frame_id = "map";
    obj.x = object.position(0);
    obj.y = object.position(1);
    obj.z = object.position(2);
    obj.label = object.label;

    res.objects.push_back(obj);
  }

  return true;
}

bool ObjectCloudNode::get_bounding_boxes(
    object_cloud::GetBoundingBoxes::Request &req,
    object_cloud::GetBoundingBoxes::Response &res) {
  geometry_msgs::TransformStamped frameToMapTransform;
  try {
    frameToMapTransform = tf_buffer.lookupTransform(
        "map", req.search_box.header.frame_id, req.search_box.header.stamp,
        ros::Duration(0.2));
  } catch (tf2::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return false;
  }
  Eigen::Affine3f frameToMap =
      tf2::transformToEigen(frameToMapTransform).cast<float>();
  Eigen::Vector3f origin(req.search_box.origin.x, req.search_box.origin.y,
                         req.search_box.origin.z);
  Eigen::Vector3f scale(req.search_box.scale.x, req.search_box.scale.y,
                        req.search_box.scale.z);

  std::vector<Object> objects =
      object_cloud.search_box(origin, scale, frameToMap);

  visualization_msgs::MarkerArray boxes;
  for (const auto &object : objects) {
    auto mit = bounding_boxes.find(object.id);

    if (mit != bounding_boxes.end()) {
      boxes.markers.push_back(mit->second);
      boxes.markers[boxes.markers.size() - 1].text = object.label;
    }
  }

  res.bounding_boxes = boxes;

  return true;
}

bool ObjectCloudNode::get_surfaces(object_cloud::GetSurfaces::Request &req,
                                   object_cloud::GetSurfaces::Response &res) {
  // Get transforms
  geometry_msgs::TransformStamped camToMapTransform;
  geometry_msgs::TransformStamped camToBaseTransform;
  geometry_msgs::TransformStamped baseToHeadPanTransform;
  try {
    camToMapTransform = tf_buffer.lookupTransform(
        "map", "head_rgbd_sensor_rgb_frame", ros::Time(0), ros::Duration(0.1));
    camToBaseTransform =
        tf_buffer.lookupTransform("base_link", "head_rgbd_sensor_rgb_frame",
                                  ros::Time(0), ros::Duration(0.1));
    baseToHeadPanTransform = tf_buffer.lookupTransform(
        "head_pan_link", "base_link", ros::Time(0), ros::Duration(0.1));
  } catch (tf2::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return false;
  }

  // Eigenify tf transforms
  Eigen::Affine3f camToMap =
      tf2::transformToEigen(camToMapTransform).cast<float>();
  Eigen::Affine3f camToBase =
      tf2::transformToEigen(camToBaseTransform).cast<float>();
  Eigen::Affine3f baseToHeadPan =
      tf2::transformToEigen(baseToHeadPanTransform).cast<float>();

  // Basically head pan link dropped down to base
  Eigen::Affine3f camToStraightCam = baseToHeadPan.rotation() * camToBase;

  // Start a point cloud request
  std::shared_ptr<PointCloudRequest> cloudReq =
      std::make_shared<PointCloudRequest>();
  cloudReq->xbounds = Eigen::Vector2f(0., 2.0);
  cloudReq->ybounds = Eigen::Vector2f(-req.view_horizon, req.view_horizon);
  cloudReq->zbounds =
      Eigen::Vector2f(0.2, std::numeric_limits<float>::infinity());
  cloudReq->cam_to_target = camToStraightCam;
  point_cloud_requests.push(cloudReq);

  std::cout << "WAITING FOR POINT CLOUD" << std::endl;
  // Wait for point cloud
  std::unique_lock<std::mutex> lock(cloudReq->mutex);
  cloudReq->cond_var.wait(lock,
                          [cloudReq] { return cloudReq->result != nullptr; });
  lock.unlock();
  std::cout << "DONE" << std::endl;

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  for (int surfaceId = 0; surfaceId < req.z_values.size(); surfaceId++) {
    // Inliers for each surface
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    // Add points that have appropriate zs for each surface
    for (int i = 0; i < cloudReq->result->points.size(); ++i) {
      // If z_values[surfaceId] is negative, we can assume user doesn't want z
      // restrictions
      if (req.z_values[surfaceId] >= 0 &&
          std::abs(cloudReq->result->points[i].z - req.z_values[surfaceId]) >
              req.z_eps) {
        continue;
      }

      inliers->indices.push_back(i);
    }

    // Extract relevant points according to appropriate zs
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
        new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloudReq->result);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*filtered);

    Eigen::Affine3f transform =
        tf2::transformToEigen(req.transforms[surfaceId]).cast<float>();
    PointCloudT::Ptr transformed(new PointCloudT);
    pcl::transformPointCloud(*filtered, *transformed, transform);

    // boost::optional<visualization_msgs::Marker> surface =
    // PointCloudUtils::extractSurface(cloudReq->result, camToMap,
    // camToStraightCam, req.x_scale, req.y_scale);
    boost::optional<visualization_msgs::Marker> surface =
        PointCloudUtils::extractSurface(
            transformed, camToMap, transform * camToStraightCam,
            req.x_scales[surfaceId], req.y_scales[surfaceId],
            req.z_scales[surfaceId], surfacecloud_pub);

    if (!surface) {
      res.surfaces.markers.emplace_back();
      continue;
    }

    res.surfaces.markers.push_back(surface.get());
    visualization_msgs::Marker &marker =
        res.surfaces.markers[res.surfaces.markers.size() - 1];
    marker.id = surfaceId;

    // The following code removes detected surfaces from the overall cloud
    // But this could make detecting next surface harder...
    /*
    // Which indices in transformed are on the surface (i.e. points with same z)
    std::vector<int> surfaceIdxs;
    float zval = ((camToMap * (transform *
    camToStraightCam).inverse()).inverse() *
                  Eigen::Vector3f(marker.pose.position.x,
    marker.pose.position.y, marker.pose.position.z))(2) +
                 req.z_scales[surfaceId] / 2.;
    for (int i = 0; i < transformed->points.size(); ++i) {
        if (std::abs(transformed->points[i].z - zval) > req.z_eps) {
            continue;
        }

        surfaceIdxs.push_back(i);
    }

    // Figure out which indices in main point cloud to remove
    pcl::PointIndices::Ptr surface_inliers(new pcl::PointIndices());
    surface_inliers->indices.reserve(surfaceIdxs.size());
    for (int i = 0; i < surfaceIdxs.size(); i++) {
        surface_inliers->indices.push_back(inliers->indices[surfaceIdxs[i]]);
    }

    // Filter out the surface
    extract.setInputCloud(cloudReq->result);
    extract.setIndices(surface_inliers);
    extract.setNegative(true);
    extract.filter(*(cloudReq->result));
    */
  }

#ifdef VISUALIZE
  surface_marker_pub.publish(res.surfaces);
#endif

  std::cout << "SUCCESS" << std::endl;

  return true;
}

bool ObjectCloudNode::get_surface_occupancy(
    object_cloud::GetSurfaceOccupancy::Request &req,
    object_cloud::GetSurfaceOccupancy::Response &res) {
  // Get all bounding boxes
  std::vector<visualization_msgs::Marker> objects;
  std::transform(
      bounding_boxes.begin(), bounding_boxes.end(), std::back_inserter(objects),
      boost::bind(
          &std::unordered_map<int,
                              visualization_msgs::Marker>::value_type::second,
          _1));

  // Compute surface occupancy intervals
  std::vector<Eigen::Vector2f> occupancy = PointCloudUtils::surfaceOccupancy(
      objects, req.surface, Eigen::Vector2f(req.x_bounds[0], req.x_bounds[1]),
      Eigen::Vector2f(req.y_bounds[0], req.y_bounds[1]));

  for (const auto &interval : occupancy) {
    res.intervals.push_back(interval(0));
    res.intervals.push_back(interval(1));
  }

  return true;
}

void ObjectCloudNode::visualize() {
  visualization_msgs::MarkerArray marker_array;
  int id = 0;
  for (const auto &object : object_cloud.get_all_objects()) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.ns = object.label;
    marker.id = id++;
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
    text_marker.id = id++;
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

    marker_array.markers.push_back(marker);
    marker_array.markers.push_back(text_marker);
  }
  viz_pub.publish(marker_array);
}
