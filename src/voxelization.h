#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <cuda.h>
#include "sceneStructs.h"

//Declare voxelization resolution (TODO: input these as a parameter)
const int log_N = 8;
const int log_T = 3; ///Can only be > 3 when in BIT_MODE
const int N = 1 << log_N; //N is the total number of voxels (per dimension)
const int M = 1 << (log_N - log_T); //M is the total number of tiles (per dimension)
const int T = 1 << log_T; //T is the tile size - voxels per tile (per dimension)

//Create bounding box to perform voxelization within
const float world_size = 2.5f; //TODO: get this from the bounding box of the input mesh
const float3 bbox0 = make_float3(-world_size, -world_size, -world_size);
const float3 bbox1 = make_float3(world_size, world_size, world_size);
const float CUBE_MESH_SCALE = 0.1f;

//Compute the 1/2 edge length for the resulting voxelization
const float vox_size = world_size / float(N);

//Compute tile/grid sizes
const float3 t_d = make_float3((bbox1.x - bbox0.x) / float(M),
  (bbox1.y - bbox0.y) / float(M),
  (bbox1.z - bbox0.z) / float(M));
const float3 p_d = make_float3(t_d.x / float(T),
  t_d.y / float(T), t_d.z / float(T));

__device__ inline float3 getCenterFromIndex(int idx, int M, int T, float3 bbox0, float3 t_d, float3 p_d) {
  int T3 = T*T*T;
  int tile_num = idx / T3;
  int pix_num = idx % T3;
  float3 cent;
  int tx = tile_num / (M*M) % M;
  int px = pix_num / (T*T) % T;
  int ty = tile_num / M % M;
  int py = pix_num / T % T;
  int tz = tile_num % M;
  int pz = pix_num % T;
  cent.x = bbox0.x + tx*t_d.x + px*p_d.x + p_d.x/2.0f;
  cent.y = bbox0.y + ty*t_d.y + py*p_d.y + p_d.y/2.0f;
  cent.z = bbox0.z + tz*t_d.z + pz*p_d.z + p_d.z/2.0f;
  return cent;
}

__host__ int voxelizeMesh(Mesh &m_in, bmp_texture* h_tex, int* d_voxels, int* d_values);

__host__ void extractCubesFromVoxelGrid(int* d_voxels, int numVoxels, int* d_values, Mesh &m_cube, Mesh &m_out);

__host__ void voxelizeToCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out);

#endif ///VOXELIZATION_H
