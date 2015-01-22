#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include <cuda.h>

#include <octree_slam/common_types.h>

namespace octree_slam {

namespace voxelization {

const float CUBE_MESH_SCALE = 0.1f;

//Declare voxelization resolution
#define GRID_RES 8
#define TILE_SIZE 3
const int log_N = GRID_RES;
const int log_T = TILE_SIZE;
const int N = 1 << log_N; //N is the total number of voxels (per dimension)
const int M = 1 << (log_N - log_T); //M is the total number of tiles (per dimension)
const int T = 1 << log_T; //T is the tile size - voxels per tile (per dimension)

struct gridParams {

  float vox_size;
  float3 bbox0, bbox1, t_d, p_d;

};

__device__ inline float3 getCenterFromIndex(int idx, int M, int T, float3 bbox0, float3 t_d, float3 p_d) {
  int T3 = T*T*T;
  int tile_num = idx / T3;
  int pix_num = idx % T3;
  float3 cent;
  int tz = tile_num / (M*M) % M;
  int pz = pix_num / (T*T) % T;
  int ty = tile_num / M % M;
  int py = pix_num / T % T;
  int tx = tile_num % M;
  int px = pix_num % T;
  cent.x = bbox0.x + tx*t_d.x + px*p_d.x + p_d.x/2.0f;
  cent.y = bbox0.y + ty*t_d.y + py*p_d.y + p_d.y/2.0f;
  cent.z = bbox0.z + tz*t_d.z + pz*p_d.z + p_d.z/2.0f;
  return cent;
}

__host__ int voxelizeMesh(Mesh &m_in, bmp_texture* h_tex, int* d_voxels, int* d_values);

__host__ void extractCubesFromVoxelGrid(int* d_voxels, int numVoxels, int* d_values, Mesh &m_cube, Mesh &m_out);

__host__ void voxelizeToCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out);

__host__ void setWorldSize(float minx, float miny, float minz, float maxx, float maxy, float maxz);

__host__ gridParams getParams();

} // namespace voxelization

} // namespace octree_slam

#endif ///VOXELIZATION_H_
