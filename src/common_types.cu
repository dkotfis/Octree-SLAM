
// CUDA Dependencies
#include <cuda.h>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

RawFrame::RawFrame(const int w, const int h) :
width(w), height(h) {
  cudaMalloc((void**)&color, h*w*sizeof(Color256));
  cudaMalloc((void**)&depth, h*w*sizeof(uint16_t));
}

RawFrame::~RawFrame() {
  cudaFree(color);
  cudaFree(depth);
}
