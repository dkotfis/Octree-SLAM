#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include <cuda_runtime.h>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace voxelization {

const float CUBE_MESH_SCALE = 0.1f;

//Declare voxelization resolution
#define GRID_RES 8
#define TILE_SIZE 3

extern "C" void voxelGridToMesh(const VoxelGrid& grid, const Mesh &m_cube, Mesh &m_out);

extern "C" void meshToVoxelGrid(const Mesh &m_in, const bmp_texture* tex, VoxelGrid &grid_out);

} // namespace voxelization

} // namespace octree_slam

#endif ///VOXELIZATION_H_
