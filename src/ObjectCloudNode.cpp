#include <geometry_msgs/Point.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>

#include <knowledge_representation/LTMCConcept.h>
#include <knowledge_representation/LTMCEntity.h>
#include <knowledge_representation/convenience.h>

#include <object_cloud/ObjectCloudNode.h>
#include <object_cloud/PointCloudConstructor.h>
#include <object_cloud/PointCloudUtils.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_ros/conversions.h>
#include <opencv/highgui.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <octomap_msgs/conversions.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <utility>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

ObjectCloudNode::ObjectCloudNode(ros::NodeHandle node, const Eigen::Matrix3f& camera_intrinsics)
  : node(node)
  , camera_intrinsics(camera_intrinsics)
  , tf_listener(tf_buffer)
  , it(node)
  , octree(0.01)
  , ltmc(knowledge_rep::getDefaultLTMC())
  , object_cloud(camera_intrinsics)
{
#ifdef VISUALIZE
  viz_detections_pub = it.advertise("detections", 1);
  viz_pub = node.advertise<visualization_msgs::MarkerArray>("markers", 1, true);
  cloud_pub = node.advertise<octomap_msgs::Octomap>("cloud", 1);
  surfacecloud_pub = node.advertise<PointCloudT>("surfacecloud", 1);
  surface_marker_pub = node.advertise<visualization_msgs::MarkerArray>("surfacemarker", 1);
  bbox_pub = node.advertise<visualization_msgs::MarkerArray>("box", 1);

  std::thread([this]() {
    while (ros::ok())
    {
      auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
      this->visualize();
      std::this_thread::sleep_until(x);
    }
  }).detach();
#endif

  // NOTE: We assume there's only one ObjectCloud, so this will blow away
  // anything that is sensed
  ltmc.getConcept("sensed").removeInstances();
  ltmc.getConcept("scanned").removeReferences();
}

ObjectCloudNode::~ObjectCloudNode()
{
  ltmc.getConcept("sensed").removeInstances();
  ltmc.getConcept("scanned").removeReferences();
}

void ObjectCloudNode::advertiseServices()
{
  clear_octree_server = node.advertiseService("clear_octree", &ObjectCloudNode::clearOctree, this);
  get_entities_server = node.advertiseService("get_entities", &ObjectCloudNode::getEntities, this);
  get_bounding_boxes_server = node.advertiseService("get_bounding_boxes", &ObjectCloudNode::getBoundingBoxes, this);
  get_objects_server = node.advertiseService("get_objects", &ObjectCloudNode::getObjects, this);
  get_surface_server = node.advertiseService("get_surfaces", &ObjectCloudNode::getSurfaces, this);
  surface_occupancy_server =
      node.advertiseService("get_surface_occupancy", &ObjectCloudNode::getSurfaceOccupancy, this);
}

void ObjectCloudNode::addToLtmc(const Object& object)
{
  auto concept = ltmc.getConcept(object.label);
  auto entity = concept.createInstance();
  auto sensed = ltmc.getConcept("sensed");
  entity.makeInstanceOf(sensed);
  entity_id_to_object.insert({ entity.entity_id, object });
}

bool ObjectCloudNode::getEntities(object_cloud::GetEntities::Request& req, object_cloud::GetEntities::Response& res)
{
  std::vector<geometry_msgs::Point> map_locations;
  for (int eid : req.entity_ids)
  {
    geometry_msgs::Point p;
    // Requested an ID that's not in the cloud. Return NANs
    if (entity_id_to_object.count(eid) == 0)
    {
      knowledge_rep::Entity entity = { static_cast<uint>(eid), ltmc };
      entity.removeAttribute("sensed");
      p.x = NAN;
      p.y = NAN;
      p.z = NAN;
    }
    else
    {
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

bool ObjectCloudNode::clearOctree(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res)
{
  std::lock_guard<std::mutex> lock(global_mutex);

  octree.clear();
  bounding_boxes.clear();
  return true;
}

bool ObjectCloudNode::getObjects(object_cloud::GetObjects::Request& req, object_cloud::GetObjects::Response& res)
{
  geometry_msgs::TransformStamped frameToMapTransform;
  try
  {
    frameToMapTransform = tf_buffer.lookupTransform("map", req.search_box.header.frame_id, req.search_box.header.stamp,
                                                    ros::Duration(0.1));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }
  Eigen::Affine3f frameToMap = tf2::transformToEigen(frameToMapTransform).cast<float>();
  Eigen::Vector3f origin(req.search_box.origin.x, req.search_box.origin.y, req.search_box.origin.z);
  Eigen::Vector3f scale(req.search_box.scale.x, req.search_box.scale.y, req.search_box.scale.z);

  std::vector<Object> objects = object_cloud.searchBox(origin, scale, frameToMap);

  for (const auto& object : objects)
  {
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

bool ObjectCloudNode::getBoundingBoxes(object_cloud::GetBoundingBoxes::Request& req,
                                       object_cloud::GetBoundingBoxes::Response& res)
{
  geometry_msgs::TransformStamped frameToMapTransform;
  try
  {
    frameToMapTransform = tf_buffer.lookupTransform("map", req.search_box.header.frame_id, req.search_box.header.stamp,
                                                    ros::Duration(0.2));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }
  Eigen::Affine3f frameToMap = tf2::transformToEigen(frameToMapTransform).cast<float>();
  Eigen::Vector3f origin(req.search_box.origin.x, req.search_box.origin.y, req.search_box.origin.z);
  Eigen::Vector3f scale(req.search_box.scale.x, req.search_box.scale.y, req.search_box.scale.z);

  std::vector<Object> objects = object_cloud.searchBox(origin, scale, frameToMap);

  visualization_msgs::MarkerArray boxes;
  for (const auto& object : objects)
  {
    auto mit = bounding_boxes.find(object.id);

    if (mit != bounding_boxes.end())
    {
      boxes.markers.push_back(mit->second);
      boxes.markers[boxes.markers.size() - 1].text = object.label;
    }
  }

  res.bounding_boxes = boxes;

  return true;
}

bool ObjectCloudNode::getSurfaces(object_cloud::GetSurfaces::Request& req, object_cloud::GetSurfaces::Response& res)
{
  // Get transforms
  geometry_msgs::TransformStamped camToMapTransform;
  geometry_msgs::TransformStamped camToBaseTransform;
  geometry_msgs::TransformStamped baseToHeadPanTransform;
  try
  {
    camToMapTransform =
        tf_buffer.lookupTransform("map", "head_rgbd_sensor_rgb_frame", ros::Time(0), ros::Duration(0.1));
    camToBaseTransform =
        tf_buffer.lookupTransform("base_link", "head_rgbd_sensor_rgb_frame", ros::Time(0), ros::Duration(0.1));
    baseToHeadPanTransform = tf_buffer.lookupTransform("head_pan_link", "base_link", ros::Time(0), ros::Duration(0.1));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }

  // Eigenify tf transforms
  Eigen::Affine3f camToMap = tf2::transformToEigen(camToMapTransform).cast<float>();
  Eigen::Affine3f camToBase = tf2::transformToEigen(camToBaseTransform).cast<float>();
  Eigen::Affine3f baseToHeadPan = tf2::transformToEigen(baseToHeadPanTransform).cast<float>();

  // Basically head pan link dropped down to base
  Eigen::Affine3f camToStraightCam = baseToHeadPan.rotation() * camToBase;

  // Start a point cloud request
  std::shared_ptr<PointCloudRequest> cloudReq = std::make_shared<PointCloudRequest>();
  cloudReq->xbounds = Eigen::Vector2f(0., 2.0);
  cloudReq->ybounds = Eigen::Vector2f(-req.view_horizon, req.view_horizon);
  cloudReq->zbounds = Eigen::Vector2f(0.2, std::numeric_limits<float>::infinity());
  cloudReq->cam_to_target = camToStraightCam;
  point_cloud_requests.push(cloudReq);

  std::cout << "WAITING FOR POINT CLOUD" << std::endl;
  // Wait for point cloud
  std::unique_lock<std::mutex> lock(cloudReq->mutex);
  cloudReq->cond_var.wait(lock, [cloudReq] { return cloudReq->result != nullptr; });
  lock.unlock();
  std::cout << "DONE" << std::endl;

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  for (int surfaceId = 0; surfaceId < req.z_values.size(); surfaceId++)
  {
    // Inliers for each surface
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    // Add points that have appropriate zs for each surface
    for (int i = 0; i < cloudReq->result->points.size(); ++i)
    {
      // If z_values[surfaceId] is negative, we can assume user doesn't want z
      // restrictions
      if (req.z_values[surfaceId] >= 0 && std::abs(cloudReq->result->points[i].z - req.z_values[surfaceId]) > req.z_eps)
      {
        continue;
      }

      inliers->indices.push_back(i);
    }

    // Extract relevant points according to appropriate zs
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloudReq->result);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*filtered);

    Eigen::Affine3f transform = tf2::transformToEigen(req.transforms[surfaceId]).cast<float>();
    PointCloudT::Ptr transformed(new PointCloudT);
    pcl::transformPointCloud(*filtered, *transformed, transform);

    // boost::optional<visualization_msgs::Marker> surface =
    // PointCloudUtils::extractSurface(cloudReq->result, camToMap,
    // camToStraightCam, req.x_scale, req.y_scale);
    boost::optional<visualization_msgs::Marker> surface =
        PointCloudUtils::extractSurface(transformed, camToMap, transform * camToStraightCam, req.x_scales[surfaceId],
                                        req.y_scales[surfaceId], req.z_scales[surfaceId], surfacecloud_pub);

    if (!surface)
    {
      res.surfaces.markers.emplace_back();
      continue;
    }

    res.surfaces.markers.push_back(surface.get());
    visualization_msgs::Marker& marker = res.surfaces.markers[res.surfaces.markers.size() - 1];
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

bool ObjectCloudNode::getSurfaceOccupancy(object_cloud::GetSurfaceOccupancy::Request& req,
                                          object_cloud::GetSurfaceOccupancy::Response& res)
{
  // Get all bounding boxes
  std::vector<visualization_msgs::Marker> objects;
  std::transform(bounding_boxes.begin(), bounding_boxes.end(), std::back_inserter(objects),
                 boost::bind(&std::unordered_map<int, visualization_msgs::Marker>::value_type::second, _1));

  // Compute surface occupancy intervals
  std::vector<Eigen::Vector2f> occupancy =
      PointCloudUtils::surfaceOccupancy(objects, req.surface, Eigen::Vector2f(req.x_bounds[0], req.x_bounds[1]),
                                        Eigen::Vector2f(req.y_bounds[0], req.y_bounds[1]));

  for (const auto& interval : occupancy)
  {
    res.intervals.push_back(interval(0));
    res.intervals.push_back(interval(1));
  }

  return true;
}

void ObjectCloudNode::visualize()
{
  std::lock_guard<std::mutex> lock(global_mutex);
// This takes time so disable if not debugging...
#if (VISUALIZE_OCTREE)
  // Publish Octree
  if (cloud_pub.getNumSubscribers() > 0)
  {
    octomap_msgs::Octomap cloud_msg;
    cloud_msg.header.frame_id = "map";
    octomap_msgs::binaryMapToMsg(octree, cloud_msg);
    cloud_pub.publish(cloud_msg);
  }
#endif

  // Publish bounding boxes
  visualization_msgs::MarkerArray boxes;
  int id = 0;
  for (const auto& e : bounding_boxes)
  {
    visualization_msgs::Marker box = e.second;
    box.id = id;
    boxes.markers.push_back(box);
    id++;
  }
  bbox_pub.publish(boxes);

  visualization_msgs::MarkerArray marker_array;
  for (const auto& object : object_cloud.getAllObjects())
  {
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

void ObjectCloudNode::updateBoundingBoxes(std::vector<std::pair<ImageBoundingBox, Object>> detection_objects,
                                          const Eigen::Affine3f& cam_to_map)
{
  // Update bounding boxes
  for (const auto& d : detection_objects)
  {
    octomap::point3d point(d.second.position(0), d.second.position(1), d.second.position(2));
    visualization_msgs::Marker box =
        PointCloudUtils::extractBoundingBox(octree, point, d.first, cam_to_map, cam_to_map, camera_intrinsics);

    // Invalid box
    if (box.header.frame_id.empty())
    {
      continue;
    }

    int key = d.second.id;
    auto mit = bounding_boxes.find(key);
    if (mit != bounding_boxes.end())
    {
      bounding_boxes.at(key) = box;
    }
    else
    {
      bounding_boxes.insert({ key, box });
    }
  }
}

void ObjectCloudNode::processPointCloudRequests(cv_bridge::CvImagePtr depth)
{
  float inf = std::numeric_limits<float>::infinity();
  while (!point_cloud_requests.empty())
  {
    std::shared_ptr<PointCloudRequest> req = point_cloud_requests.pop();
    std::cout << "PROCESSING" << std::endl;
    {
      std::lock_guard<std::mutex> lock(req->mutex);

      octomap::Pointcloud planecloud = PointCloudConstructor::construct(
          camera_intrinsics, depth->image, req->cam_to_target, inf, req->xbounds, req->ybounds, req->zbounds);
      pcl::PointCloud<pcl::PointXYZ>::Ptr req_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      req_cloud->points.reserve(planecloud.size());
      for (const auto& p : planecloud)
      {
        req_cloud->points.emplace_back(p.x(), p.y(), p.z());
      }
      req->result = req_cloud;
    }
    std::cout << "PROCESSED" << std::endl;

    req->cond_var.notify_one();
    std::cout << "NOTIFIED" << std::endl;
  }
}
