#ifndef SVO_H_
#define SVO_H_

#include <cuda_runtime.h>

#include <octree_slam/common_types.h>

#define USE_BRICK_POOL false

namespace octree_slam {

namespace svo {

extern "C" void svoFromVoxelGrid(const VoxelGrid& grid, const int max_depth, int* &d_octree, int& octree_size, glm::vec3 octree_center, const float edge_length, cudaArray* d_bricks = NULL);

extern "C" void extractVoxelGridFromSVO(int* &d_octree, int& octree_size, const int max_depth, const glm::vec3 center, float edge_length, VoxelGrid& grid);

inline __host__ __device__ int oppositeNode(const int node) {
  //Returns the bitwise complement
  return -(~node);
}

} // namespace svo

} // namespace octree_slam

#endif //SVO_H_
