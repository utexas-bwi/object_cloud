#include <object_cloud/ObjectCloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <utility>
#include <limits>
#include <vector>

using cv::Mat;
using nanoflann::RadiusResultSet;
using std::abs;
using std::make_pair;
using std::pair;
using std::vector;

const float NEIGHBOR_THRESH_X = 0.2;
const float NEIGHBOR_THRESH_Y = 0.2;
const float NEIGHBOR_THRESH_Z = 0.3;

/*
 * Adds object
 * Returns (new object?, 3D point)
 */
pair<bool, Object> ObjectCloud::addObject(const ImageBoundingBox& bbox, const Mat& rgb_image, const Mat& depth_image,
                                          const Eigen::Affine3f& cam_to_map)
{
  if (bbox.y + bbox.height > rgb_image.rows || bbox.x + bbox.width > rgb_image.cols)
  {
    // std::cout << bbox.x << std::endl;
    // std::cout << bbox.y << std::endl;
    // std::cout << bbox.width << std::endl;
    // std::cout << bbox.height << std::endl;
    // throw std::runtime_error("Invalid bbox");
    return make_pair(false, Object());
  }

  // Get the depth of the object
  // We could use GrabCut, but it takes too long ~5fps
  // So just get the depth of the center pixel
  // It's probably good enough ¯\_(ツ)_/¯
  float depth = depth_image.at<uint16_t>(bbox.y + bbox.height / 2., bbox.x + bbox.width / 2.) / 1000.;
  if (depth == 0 || isnan(depth))
  {
    return make_pair(false, Object());
  }

  float intrinsic_cx = this->camera_intrinsics(0, 2);
  float intrinsic_cy = this->camera_intrinsics(1, 2);
  float intrinsic_sx = this->camera_intrinsics(0, 0);
  float intrinsic_sy = this->camera_intrinsics(1, 1);

  // Compute 3d point in the world
  float x_center = bbox.x + bbox.width / 2.;
  float y_center = bbox.y + bbox.height / 2.;
  float camera_x = (x_center - intrinsic_cx) * (1.0 / intrinsic_sx) * depth;
  float camera_y = (y_center - intrinsic_cy) * (1.0 / intrinsic_sy) * depth;
  float camera_z = depth;
  Eigen::Vector3f world = cam_to_map * Eigen::Vector3f(camera_x, camera_y, camera_z);

  // Check if point exists in KDTree
  float candidatePoint[3] = { world(0), world(1), world(2) };

  // We want to search the smallest ball that contains our threshold box
  if (!cloud.pts.empty())
  {
    float sqRadius = NEIGHBOR_THRESH_X * NEIGHBOR_THRESH_X + NEIGHBOR_THRESH_Y * NEIGHBOR_THRESH_Y +
                     NEIGHBOR_THRESH_Z * NEIGHBOR_THRESH_Z;
    std::vector<std::pair<size_t, float>> indices_dists;
    RadiusResultSet<float, size_t> resultSet(sqRadius, indices_dists);
    cloud_index.findNeighbors(resultSet, candidatePoint, nanoflann::SearchParams());

    for (const auto& e : indices_dists)
    {
      PointCloud::Point& p = cloud.pts[e.first];
      if (abs(candidatePoint[0] - p.x) <= NEIGHBOR_THRESH_X && abs(candidatePoint[1] - p.y) <= NEIGHBOR_THRESH_Y &&
          abs(candidatePoint[2] - p.z) <= NEIGHBOR_THRESH_Z)
      {
        // Same label?
        auto mit = objects_data.find(p.id);
        if (mit != objects_data.end() && mit->second.label == bbox.label)
        {
          return make_pair(false, mit->second);
        }
      }
    }
  }

  // If it doesn't exist, we can add it to our cloud
  cloud.add(world(0), world(1), world(2));
  int idx = cloud.pts.size() - 1;
  cloud_index.addPoints(idx, idx);  // [start, end] inclusive
  Object object;
  object.position = Eigen::Vector3f(world(0), world(1), world(2));
  object.label = bbox.label;
  object.id = cloud.pts[idx].id;
  objects_data.insert({ cloud.pts[idx].id, object });

  return make_pair(true, object);
}

/*
 * Adds ground truth object
 * Returns (new object?, 3D point)
 */
pair<bool, Object> ObjectCloud::addObject(Object& object)
{
  float candidatePoint[3] = { object.position(0), object.position(1), object.position(2) };

  // We want to search the smallest ball that contains our threshold box
  if (!cloud.pts.empty())
  {
    float sqRadius = NEIGHBOR_THRESH_X * NEIGHBOR_THRESH_X + NEIGHBOR_THRESH_Y * NEIGHBOR_THRESH_Y +
                     NEIGHBOR_THRESH_Z * NEIGHBOR_THRESH_Z;
    std::vector<std::pair<size_t, float>> indices_dists;
    RadiusResultSet<float, size_t> resultSet(sqRadius, indices_dists);
    cloud_index.findNeighbors(resultSet, candidatePoint, nanoflann::SearchParams());

    for (const auto& e : indices_dists)
    {
      PointCloud::Point& p = cloud.pts[e.first];
      if (std::abs(candidatePoint[0] - p.x) <= NEIGHBOR_THRESH_X &&
          std::abs(candidatePoint[1] - p.y) <= NEIGHBOR_THRESH_Y &&
          std::abs(candidatePoint[2] - p.z) <= NEIGHBOR_THRESH_Z)
      {
        // Same label?
        auto mit = objects_data.find(p.id);
        if (mit != objects_data.end() && mit->second.label == object.label)
        {
          return make_pair(false, mit->second);
        }
      }
    }
  }

  // If it doesn't exist, we can add it to our cloud
  cloud.add(object.position(0), object.position(1), object.position(2));
  int idx = cloud.pts.size() - 1;
  cloud_index.addPoints(idx, idx);  // [start, end] inclusive
  object.id = cloud.pts[idx].id;
  objects_data.insert({ cloud.pts[idx].id, object });

  return std::make_pair(true, object);
}

vector<Object> ObjectCloud::getAllObjects()
{
  vector<Object> objects;
  objects.reserve(objects_data.size());

  for (const auto& e : objects_data)
  {
    objects.push_back(e.second);
  }

  return objects;
}

vector<Object> ObjectCloud::searchBox(const Eigen::Vector3f& origin, const Eigen::Vector3f& scale,
                                      const Eigen::Affine3f& box_to_map)
{
  vector<Object> objects;

  Eigen::Vector3f min(origin(0) - scale(0), origin(1) - scale(1), origin(2) - scale(2));
  Eigen::Vector3f max(origin(0) + scale(0), origin(1) + scale(1), origin(2) + scale(2));
  Eigen::Vector3f center_pt = box_to_map * origin;
  Eigen::Affine3f mapToBox = box_to_map.inverse();

  // We want to search the smallest ball that contains our threshold box
  float sqRadius = (max - min).squaredNorm();
  float center[3] = { center_pt(0), center_pt(1), center_pt(2) };
  std::vector<std::pair<size_t, float>> indices_dists;
  RadiusResultSet<float, size_t> resultSet(sqRadius, indices_dists);
  cloud_index.findNeighbors(resultSet, center, nanoflann::SearchParams());

  for (const auto& e : indices_dists)
  {
    PointCloud::Point& p = cloud.pts[e.first];
    Eigen::Vector3f point = mapToBox * Eigen::Vector3f(p.x, p.y, p.z);

    if (point(0) >= min(0) && point(0) <= max(0) && point(1) >= min(1) && point(1) <= max(1) && point(2) >= min(2) &&
        point(2) <= max(2))
    {
      auto mit = objects_data.find(p.id);
      if (mit != objects_data.end())
      {
        objects.push_back(mit->second);
      }
    }
  }

  return objects;
}

void ObjectCloud::clear()
{
  for (int idx = 0; idx < cloud.pts.size(); idx++)
  {
    cloud_index.removePoint(idx);
  }
  cloud.clear();
}
