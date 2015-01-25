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

//Computes ICP cost terms from GPU vertex/normal/color maps and fills in matrices in CPU memory
extern "C" void computeICPCost(const Frame& last_frame, const Frame& this_frame, float* A, float* b);

//Computes RGBD cost terms from GPU vertex/normal/color maps and fills in matrices in CPU memory
extern "C" void computeRGBDCost(const Frame& last_frame, const Frame& this_frame, float* A, float* b);

} // namespace sensor

} // namespace octree_slam

#endif //LOCALIZATION_KERNELS_H_
