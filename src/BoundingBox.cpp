#include <BoundingBox.h>
#include "Eigen/Dense"

BoundingBox::BoundingBox(const Eigen::Vector4f &min, const Eigen::Vector4f &max, const Eigen::Vector4f &centroid, const Eigen::Quaternionf &orientation, const Eigen::Vector4f &position, const int index):
			min(min), max(max),
			centroid(centroid), position(position), orientation(orientation),
			index(index){

}
