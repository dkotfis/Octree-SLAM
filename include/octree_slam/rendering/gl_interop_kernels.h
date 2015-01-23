#ifndef GL_INTEROP_KERNELS_H_
#define GL_INTEROP_KERNELS_H_

#include <cuda_runtime.h>

#include <octree_slam/common_types.h>

namespace octree_slam {

namespace rendering {

//Function for copying point cloud data to a vbo and cbo
extern "C" void copyPointsToGL(const glm::vec3* vertices, const Color256* colors, float3* vbo, float3* cbo, const int num_points);

} // namespace rendering

} // namespace octree_slam

#endif //GL_INTEROP_KERNELS_H_
