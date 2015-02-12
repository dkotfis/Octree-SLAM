#ifndef SVO_H_
#define SVO_H_

#include <cuda_runtime.h>

#include <octree_slam/common_types.h>

#define SVO_RES 8
#define USE_BRICK_POOL false

namespace octree_slam {

namespace svo {

//Declare octree rendering resolution
const int log_SVO_N = SVO_RES;

extern "C" void svoFromVoxelGrid(VoxelGrid& grid, int* d_octree, cudaArray* d_bricks);

extern "C" void extractVoxelGridFromSVO(int* d_octree, int numVoxels, VoxelGrid& grid);

inline __host__ __device__ int oppositeNode(const int node) {
  //Returns the bitwise complement
  return ~node;
}

} // namespace svo

} // namespace octree_slam

#endif //SVO_H_
