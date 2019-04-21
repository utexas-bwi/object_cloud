#include "villa_yolocloud/PointCloudConstructor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <cmath>

using namespace cv;
using namespace Eigen;

octomap::Pointcloud PointCloudConstructor::construct(const Matrix3f &ir_intrinsics,
                                                     const Mat &depthImage,
                                                     const Eigen::Affine3f &transform) {
    Matrix<float, 4, Dynamic> points(4, depthImage.rows * depthImage.cols);

    float sx = ir_intrinsics(0, 0);
    float sy = ir_intrinsics(1, 1);
    float x_c = ir_intrinsics(0, 2);
    float y_c = ir_intrinsics(1, 2);

    float *pointsData = points.data(); // Raw data pointer for greater efficiency
    int valid_points = 0;

    for (int y = 0; y < depthImage.rows; ++y) {
        const float *row = depthImage.ptr<float>(y);

        for (int x = 0; x < depthImage.cols; ++x) {
            // If depth ain't valid, skip
            if (std::isnan(row[x]) || row[x] == 0) {
                continue;
            }

            // Analytic solution to intrinsics^-1(point) * depth
            // Eigen is column major order, so *4 is column size
            pointsData[valid_points*4 + 0] =
                    (x - x_c) * (1.0 / sx) * row[x];
            pointsData[valid_points*4 + 1] =
                    (y - y_c) * (1.0 / sy) * row[x];

            pointsData[valid_points*4 + 2] = row[x];
            pointsData[valid_points*4 + 3] = 1.0f;

            valid_points++;
        }
    }

    // Transform points
    Matrix<float, 4, Dynamic> pointsTransformed = transform * points;

    octomap::Pointcloud cloud;
    cloud.reserve(valid_points);

    // Raw data pointer for greater efficiency
    const float *pointsTransformedData = pointsTransformed.data();

    // Fill in octomap point cloud
    for (int i = 0; i < valid_points; ++i) {
        float tfx = pointsTransformedData[i*4 + 0] / pointsTransformedData[i*4 + 3];
        float tfy = pointsTransformedData[i*4 + 1] / pointsTransformedData[i*4 + 3];
        float tfz = pointsTransformedData[i*4 + 2] / pointsTransformedData[i*4 + 3];

        cloud.push_back(octomap::point3d(tfx, tfy, tfz));
    }

    return cloud;
}
