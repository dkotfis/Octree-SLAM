#ifndef RGBD_CAMERA_H_
#define RGBD_CAMERA_H_

// OpenGL Dependency
#include <glm/glm.hpp>

// Octree-SLAM Dependency
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace sensor {

// Forward Declaration
class ICPFrame;
class RGBDFrame;

class RGBDCamera {

public:

  //Default Constructor
  //Represents an RGBD camera (abstracted from the actual device)
  RGBDCamera(const int width, const int height, const glm::vec2 &focal_length);

  //Destructor
  ~RGBDCamera();

  //Gets the camera matrices for the current position/orientation
  const Camera camera() const;

  //Accessor function for the camera position
  const glm::vec3 position() const { return position_; };

  //Accessor function for the camera orientation matrix
  const glm::mat3 orientation() const { return orientation_; };

  //Updates the camera pose by predicting with a new frame
  void update(const RawFrame* this_frame);

private:

  //Utility function for solving cholesky decomposition problem
  void solveCholesky(const int dimension, const float* A, const float* b, float* x) const;

  //Camera position
  glm::vec3 position_;

  //Camera orientation
  glm::mat3 orientation_;

  //Camera view information
  glm::vec2 focal_length_;
  int width_, height_;

  //Store the previously seen frame
  ICPFrame* last_icp_frame_;
  RGBDFrame* last_rgbd_frame_;
  bool has_frame_;

  //The number of depth map pyramids to use
  static const int PYRAMID_DEPTH = 3;

  //The number of iterations for each level in the depth map pyramid;
  static const int PYRAMID_ITERS[PYRAMID_DEPTH];

  //The relative weight of the RGBD cost contribution
  static const float W_RGBD;

}; //class RGBDCamera

} //namespace sensor

} //namespace octree_slam


#endif //RGBD_CAMERA_H_

