#ifndef SVO_H_
#define SVO_H_

#include <cuda.h>

#include <octree_slam/common_types.h>

#define SVO_RES 8
#define USE_BRICK_POOL false

namespace octree_slam {

namespace svo {

//Declare octree rendering resolution
const int log_SVO_N = SVO_RES;

__host__ void svoFromVoxels(int* d_voxels, int numVoxels, int* d_values, int* d_octree, cudaArray* d_bricks);

__host__ void extractCubesFromSVO(int* d_octree, int numVoxels, Mesh &m_cube, Mesh &m_out);

__host__ void voxelizeSVOCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out);

} // namespace svo

} // namespace octree_slam

#endif //SVO_H_
