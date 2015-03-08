
// CUDA Dependencies
#include <cuda.h>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

bool BoundingBox::contains(const BoundingBox& other) const {
  bool result = true;

  result = result && (bbox0.x <= other.bbox0.x);
  result = result && (bbox0.y <= other.bbox0.y);
  result = result && (bbox0.z <= other.bbox0.z);

  result = result && (bbox1.x >= other.bbox1.x);
  result = result && (bbox1.y >= other.bbox1.y);
  result = result && (bbox1.z >= other.bbox1.z);

  return result;
}

float BoundingBox::distanceOutside(const BoundingBox& other) const {
  float result = 0.0f;

  result = max(result, other.bbox0.x - bbox0.x);
  result = max(result, other.bbox0.y - bbox0.y);
  result = max(result, other.bbox0.z - bbox0.z);

  result = max(result, bbox1.x - other.bbox1.x);
  result = max(result, bbox1.y - other.bbox1.y);
  result = max(result, bbox1.z - other.bbox1.z);

  return result;
}

RawFrame::RawFrame(const int w, const int h) :
width(w), height(h) {
  cudaMalloc((void**)&color, h*w*sizeof(Color256));
  cudaMalloc((void**)&depth, h*w*sizeof(uint16_t));
}

RawFrame::~RawFrame() {
  cudaFree(color);
  cudaFree(depth);
}

VoxelGrid::~VoxelGrid() {
  if (size > 0) {
    cudaFree(centers);
    cudaFree(colors);
  }
}

