#ifndef LOCALIZATION_KERNELS_H_
#define LOCALIZATION_KERNELS_H_

// CUDA Dependencies
#include <cuda_runtime.h>

// OpenGL Dependency
#include <glm/glm.hpp>

// Octree-SLAM Depednency
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace sensor {

struct ICPFrame {
  ICPFrame(const int w, const int h);
  ~ICPFrame();
  glm::vec3* vertex;
  glm::vec3* normal;
  int width;
  int height;
};

struct RGBDFrame {
  RGBDFrame(const int w, const int h);
  ~RGBDFrame();
  float* intensity;
  glm::vec3* vertex;
  int width;
  int height;
};

//Computes ICP cost terms from GPU vertex/normal/color maps and fills in matrices in CPU memory
extern "C" void computeICPCost(const ICPFrame* last_frame, const ICPFrame& this_frame, float* A, float* b);

//Computes ICP cost terms with a single kernel determining correspondences and computing cost
extern "C" void computeICPCost2(const ICPFrame* last_frame, const ICPFrame& this_frame, float* A, float* b);

//Computes RGBD cost terms from GPU vertex/normal/color maps and fills in matrices in CPU memory
extern "C" void computeRGBDCost(const RGBDFrame* last_frame, const RGBDFrame& this_frame, float* A, float* b);

} // namespace sensor

} // namespace octree_slam

#endif //LOCALIZATION_KERNELS_H_
