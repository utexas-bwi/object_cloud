#include <object_cloud/PointCloudConstructor.h>
#include <opencv2/core/eigen.hpp>

using cv::Mat;
using Eigen::Matrix3f;
using Eigen::Matrix;
using Eigen::Dynamic;

octomap::Pointcloud PointCloudConstructor::construct(const Matrix3f& ir_intrinsics, const Mat& depthImage,
                                                     const Eigen::Affine3f& transform, float maxDist,
                                                     const Eigen::Vector2f& xBounds, const Eigen::Vector2f& yBounds,
                                                     const Eigen::Vector2f& zBounds)
{
  Matrix<float, 3, Dynamic> points(3, depthImage.rows * depthImage.cols);

  float maxDistSq = maxDist * maxDist;

  float sx = ir_intrinsics(0, 0);
  float sy = ir_intrinsics(1, 1);
  float x_c = ir_intrinsics(0, 2);
  float y_c = ir_intrinsics(1, 2);

  float* pointsData = points.data();  // Raw data pointer for greater efficiency
  int valid_points = 0;

  for (int y = 0; y < depthImage.rows; ++y)
  {
    const uint16_t* row = depthImage.ptr<uint16_t>(y);

    for (int x = 0; x < depthImage.cols; ++x)
    {
      // If depth ain't valid, skip
      if (row[x] == 0)
      {
        continue;
      }

      float depth = row[x] / 1000.;

      // Analytic solution to intrinsics^-1(point) * depth
      // The * depth is so we pick the right representative from our equivalence
      // class
      // Eigen is column major order, so *3 is column size
      pointsData[valid_points * 3 + 0] = (x - x_c) * (1.0 / sx) * depth;
      pointsData[valid_points * 3 + 1] = (y - y_c) * (1.0 / sy) * depth;

      pointsData[valid_points * 3 + 2] = depth;

      float distSq = pointsData[valid_points * 3 + 0] * pointsData[valid_points * 3 + 0] +
                     pointsData[valid_points * 3 + 1] * pointsData[valid_points * 3 + 1] +
                     pointsData[valid_points * 3 + 2] * pointsData[valid_points * 3 + 2];
      if (distSq >= maxDistSq)
      {
        continue;
      }

      valid_points++;
    }
  }

  // Transform points
  Matrix<float, 3, Dynamic> pointsTransformed = transform * points;

  octomap::Pointcloud cloud;
  cloud.reserve(valid_points);

  // Raw data pointer for greater efficiency
  const float* pointsTransformedData = pointsTransformed.data();

  // Fill in octomap point cloud
  for (int i = 0; i < valid_points; ++i)
  {
    float x = pointsTransformedData[i * 3 + 0];
    float y = pointsTransformedData[i * 3 + 1];
    float z = pointsTransformedData[i * 3 + 2];

    if (z <= zBounds(0) || z >= zBounds(1))
    {
      continue;
    }
    if (y <= yBounds(0) || y >= yBounds(1))
    {
      continue;
    }
    if (x <= xBounds(0) || x >= xBounds(1))
    {
      continue;
    }

    cloud.push_back(octomap::point3d(x, y, z));
  }

  return cloud;
}
