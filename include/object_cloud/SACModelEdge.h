#pragma once

#include <pcl/sample_consensus/sac_model_line.h>
#include <limits>
#include <algorithm>
#include <vector>

template <typename PointT>
class SACModelEdge : public pcl::SampleConsensusModelLine<PointT>
{
public:
  typedef typename pcl::SampleConsensusModelLine<PointT>::PointCloud PointCloud;
  typedef typename pcl::SampleConsensusModelLine<PointT>::PointCloudPtr PointCloudPtr;
  typedef typename pcl::SampleConsensusModelLine<PointT>::PointCloudConstPtr PointCloudConstPtr;

  typedef boost::shared_ptr<SACModelEdge> Ptr;

  /** \brief Constructor for base SACModelCustomLine.
    * \param[in] cloud the input point cloud dataset
    * \param[in] random if true set the random seed to the current time, else
   * set to 12345 (default: false)
    */
  SACModelEdge(const PointCloudConstPtr& cloud, std::vector<Eigen::Vector3f>& invalid_lines, float min_length,
               float min_density, bool random = false)
    : invalid_lines_(invalid_lines)
    , pcl::SampleConsensusModelLine<PointT>(cloud, random)
    , axis_(Eigen::Vector3f::Zero())
    , eps_cosine_(0.0)
    , min_length_(min_length)
    , min_density_(min_density)
  {
    for (auto& line : invalid_lines)
    {
      line.normalize();
    }
  }

  /** \brief Empty destructor */
  ~SACModelEdge()
  {
  }

  /** \brief Set the axis along which we need to search for a line.
    * \param[in] ax the axis along which we need to search for a line
    */
  inline void setAxis(const Eigen::Vector3f& ax)
  {
    axis_ = ax;
    axis_.normalize();
  }

  /** \brief Get the axis along which we need to search for a line. */
  inline Eigen::Vector3f getAxis() const
  {
    return (axis_);
  }

  /** \brief Set the angle epsilon (delta) threshold.
    * \param[in] ea the maximum allowed difference between the line direction
   * and the given axis (in radians).
    */
  inline void setEpsCosine(const double ea)
  {
    eps_cosine_ = ea;
  }

  /** \brief Get the angle epsilon (delta) threshold (in radians). */
  inline double getEpsCosine() const
  {
    return (eps_cosine_);
  }

  /** \brief Return an unique id for this model (SACMODEL_PARALLEL_LINE). */
  inline pcl::SacModel getModelType() const override
  {
    return (pcl::SACMODEL_PARALLEL_LINE);
  }

  void selectWithinDistance(const Eigen::VectorXf& model_coefficients, const double threshold,
                            std::vector<int>& inliers)
  {
    // Needs a valid set of model coefficients
    if (!isModelValid(model_coefficients))
      return;

    double sqr_threshold = threshold * threshold;

    int nr_p = 0;
    inliers.resize(indices_->size());
    error_sqr_dists_.resize(indices_->size());

    // Obtain the line point and direction
    Eigen::Vector4f line_pt(model_coefficients[0], model_coefficients[1], model_coefficients[2], 0);
    Eigen::Vector4f line_dir(model_coefficients[3], model_coefficients[4], model_coefficients[5], 0);
    line_dir.normalize();

    // TODO(rishihahs): Assuming z is orthogonal axis to line. Really, we should project
    // this line properly onto the user-provided axis
    float dirs[2] = { std::abs(line_dir(0)), std::abs(line_dir(1)) };
    int min = std::min_element(dirs, dirs + 2) - dirs;
    int max = 1 - min;

    // Iterate through the 3d points and calculate the distances from them to
    // the line
    for (size_t i = 0; i < indices_->size(); ++i)
    {
      // Calculate the distance from the point to the line
      // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) /
      // norm(p2-p1)
      double sqr_distance = (line_pt - input_->points[(*indices_)[i]].getVector4fMap()).cross3(line_dir).squaredNorm();

      const Eigen::Vector3f& pt = input_->points[(*indices_)[i]].getVector3fMap();
      bool oncliff = pt(min) > (line_pt(min) + (pt(max) - line_pt(max)) / line_dir(max) * line_dir(min) - threshold);

      if ((sqr_distance < sqr_threshold) && oncliff)
      {
        // Returns the indices of the points whose squared distances are smaller
        // than the threshold
        inliers[nr_p] = (*indices_)[i];
        error_sqr_dists_[nr_p] = sqr_distance;
        ++nr_p;
      }
    }
    inliers.resize(nr_p);
    error_sqr_dists_.resize(nr_p);
  }

  int countWithinDistance(const Eigen::VectorXf& model_coefficients, const double threshold) override
  {
    // Needs a valid set of model coefficients
    if (!isModelValid(model_coefficients))
      return (0);

    double sqr_threshold = threshold * threshold;

    int nr_p = 0;

    // Obtain the line point and direction
    Eigen::Vector4f line_pt(model_coefficients[0], model_coefficients[1], model_coefficients[2], 0);
    Eigen::Vector4f line_dir(model_coefficients[3], model_coefficients[4], model_coefficients[5], 0);
    line_dir.normalize();

    // TODO(rishihahs): Assuming z is orthogonal axis to line. Really, we should project
    // this line properly onto the user-provided axis
    float dirs[2] = { std::abs(line_dir(0)), std::abs(line_dir(1)) };
    int min = std::min_element(dirs, dirs + 2) - dirs;
    int max = 1 - min;

    // Iterate through the 3d points and calculate the distances from them to
    // the line
    for (size_t i = 0; i < indices_->size(); ++i)
    {
      // Calculate the distance from the point to the line
      // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) /
      // norm(p2-p1)
      double sqr_distance = (line_pt - input_->points[(*indices_)[i]].getVector4fMap()).cross3(line_dir).squaredNorm();

      const Eigen::Vector3f& pt = input_->points[(*indices_)[i]].getVector3fMap();
      bool oncliff = pt(min) > (line_pt(min) + (pt(max) - line_pt(max)) / line_dir(max) * line_dir(min) - threshold);

      if ((sqr_distance < sqr_threshold) && oncliff)
      {
        nr_p++;
      }
    }
    std::cout << nr_p << std::endl;
    return (nr_p);
  }

protected:
  using pcl::SampleConsensusModel<PointT>::indices_;
  using pcl::SampleConsensusModel<PointT>::input_;
  using pcl::SampleConsensusModel<PointT>::error_sqr_dists_;
  using pcl::SampleConsensusModel<PointT>::max_sample_checks_;
  using pcl::SampleConsensusModel<PointT>::shuffled_indices_;

  /** \brief The axis along which we need to search for a line. */
  Eigen::Vector3f axis_;

  /** \brief The maximum allowed difference between the line direction and the
   * given axis. */
  double eps_cosine_;

  std::vector<Eigen::Vector3f>& invalid_lines_;

  float min_length_;
  float min_density_;

  /** \brief Check whether a model is valid given the user constraints.
    * \param[in] model_coefficients the set of model coefficients
    */
  bool isModelValid(const Eigen::VectorXf& model_coefficients) override
  {
    if (!pcl::SampleConsensusModelLine<PointT>::isModelValid(model_coefficients))
      return (false);

    // Obtain the line direction
    Eigen::Vector3f line_dir(model_coefficients[3], model_coefficients[4], model_coefficients[5]);
    line_dir.normalize();

    // Check against template, if given
    if (eps_cosine_ > 0.0)
    {
      float cos = std::abs(line_dir.dot(axis_));

      // Check whether the current line model satisfies our angle threshold
      // criterion with respect to the given axis
      if (cos > eps_cosine_)
        return (false);
    }

    // TODO(rishihahs): Better line comparison, for now just see if angle between
    // directions is close enough
    for (auto& line : invalid_lines_)
    {
      float cos = line_dir.dot(line);
      if (std::abs(cos) >= 0.8)
      {
        return false;
      }
    }

    // Obtain the line point and direction
    Eigen::Vector3f line_pt(model_coefficients[0], model_coefficients[1], model_coefficients[2]);

    // TODO(rishihahs): Assuming z is orthogonal axis to line. Really, we should project
    // this line properly onto the user-provided axis
    float dirs[2] = { std::abs(line_dir(0)), std::abs(line_dir(1)) };
    int min = std::min_element(dirs, dirs + 2) - dirs;
    int max = 1 - min;

    // Iterate through the 3d points and calculate the distances from them to
    // the line
    float minX = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    int nr_p = 0;
    int points_on_line = 0;
    for (size_t i = 0; i < indices_->size(); ++i)
    {
      const Eigen::Vector3f& pt = input_->points[(*indices_)[i]].getVector3fMap();
      bool oncliff = pt(min) > (line_pt(min) + (pt(max) - line_pt(max)) / line_dir(max) * line_dir(min) - 0.01);

      // Calculate the distance from the point to the line
      // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) /
      // norm(p2-p1)
      double sqr_distance = (line_pt - input_->points[(*indices_)[i]].getVector3fMap()).cross(line_dir).squaredNorm();

      if (oncliff)
      {
        nr_p++;

        float sqr_threshold = 0.01 * 0.01;
        if (sqr_distance < sqr_threshold)
        {
          points_on_line++;
          minX = std::min(minX, pt(0));
          maxX = std::max(maxX, pt(0));
          minY = std::min(minY, pt(1));
          maxY = std::max(maxY, pt(1));
        }
      }
    }
    // Percentage of points on the cliff
    float valid = static_cast<float>(nr_p) / indices_->size();
    if (valid < 0.99)
    {
      return false;
    }

    // If the length of the line is too small, then well
    // Size of line
    float length = Eigen::Vector2f(maxX - minX, maxY - minY).norm();
    if (length < min_length_)
    {
      return false;
    }

    // Density of line (ratio of points on line to length)
    // Pretend each point is a millimeter
    float pointssize = 0.001 * points_on_line;
    if (pointssize / length < min_density_)
    {
      return false;
    }

    return true;
  }
};
