#ifndef SVO_H
#define SVO_H

#include <cuda.h>
#include "sceneStructs.h"

//Declare octree rendering resolution
const int log_SVO_N = 7;

__host__ void svoFromVoxels(int* d_voxels, int numVoxels, int* d_values, int* d_octree);

__host__ void extractCubesFromSVO(int* d_octree, int numVoxels, Mesh &m_cube, Mesh &m_out);

__host__ void voxelizeSVOCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out);

#endif ///SVO_H
