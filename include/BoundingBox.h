#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include <ros/ros.h>
#include <Eigen/Dense>

class BoundingBox{
public:
	Eigen::Vector4f centroid;
	Eigen::Vector4f min;
	Eigen::Vector4f max;

	int index;

	Eigen::Quaternionf orientation;
	Eigen::Vector4f position;

	BoundingBox(const Eigen::Vector4f &min, const Eigen::Vector4f &max, const Eigen::Vector4f &centroid, const Eigen::Quaternionf &orientation, const Eigen::Vector4f &position, const int index);
};

#endif
