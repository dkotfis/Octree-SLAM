#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include <cuda_runtime.h>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace voxelization {

extern "C" void voxelizeToCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out);

extern "C" void voxelizeToGrid(Mesh &m_in, bmp_texture* tex, VoxelGrid &grid_out);

extern "C" void setWorldSize(float minx, float miny, float minz, float maxx, float maxy, float maxz);

} // namespace voxelization

} // namespace octree_slam

#endif ///VOXELIZATION_H_
