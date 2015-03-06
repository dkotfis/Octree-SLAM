
#include <cuda.h>

// Octree-SLAM Dependencies
#include <octree_slam/rendering/gl_interop_kernels.h>

namespace octree_slam {

namespace rendering {

__global__ void copyPointsToGLKernel(const glm::vec3* vertices, const Color256* colors, float3* vbo, float3* cbo, const int num_points) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= num_points) {
    return;
  }

  //Grab the point values once from global memory
  glm::vec3 pos = vertices[idx];
  Color256 color = colors[idx];

  // Each thread writes one point
  vbo[idx].x = pos.r;
  vbo[idx].y = pos.g;
  vbo[idx].z = pos.b;
  cbo[idx].x = color.r / 255.0f;
  cbo[idx].y = color.g / 255.0f;
  cbo[idx].z = color.b / 255.0f;

}

extern "C" void copyPointsToGL(const glm::vec3* vertices, const Color256* colors, float3* vbo, float3* cbo, const int num_points) {
  copyPointsToGLKernel<<<ceil((float)num_points / 256.0f), 256>>>(vertices, colors, vbo, cbo, num_points);
  cudaDeviceSynchronize();
}

} // namespace rendering

} // namespace octree_slam
