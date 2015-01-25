
// Octree-SLAM Dependencies
#include <octree_slam/sensor/openni_kernels.h>

// CUDA / OpenGL Dependencies
#include <cuda_gl_interop.h>

namespace octree_slam {

namespace sensor {

__global__ void generateVertexMapKernel(const openni::DepthPixel* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= width*height) {
    return;
  }

  //Compute the x/y coords of this thread
  int x = idx % width;
  int y = idx / width;

  //Get the depth value for this pixel from global memory once
  int depth = depth_pixels[idx];
  //TODO: Handle no-measurements

  //Conversion from millimeters to meters
  const float milli = 0.001f;

  //Compute the point coordinates
  vertex_map[idx].x = (x - width/2) * (float) depth / focal_length.x * milli;
  vertex_map[idx].y = (height/2 - y) * (float) depth / focal_length.y * milli;
  vertex_map[idx].z = depth*milli;

}

extern "C" void generateVertexMap(const openni::DepthPixel* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length) {
  generateVertexMapKernel<<<width*height / 256 + 1, 256>>>(depth_pixels, vertex_map, width, height, focal_length);
  cudaDeviceSynchronize();
}

__global__ void generateNormalMapKernel(const glm::vec3* vertex_map, glm::vec3* normal_map, const int width, const int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= width*height) {
    return;
  }

  //Compute the x/y coords of this thread
  int x = idx % width;
  int y = idx / width;

  //Get the center point from global memory once
  glm::vec3 center = vertex_map[idx];

  //Determine which direction to offset
  int offx = x > width/2 ? -1 : 1;
  int offy = y > height/2 ? -width : width;

  //Compute two vectors within the surface (locally)
  glm::vec3 v1 = vertex_map[idx + offx] - center;
  glm::vec3 v2 = vertex_map[idx + offy] - center;

  //Compute the normal
  glm::vec3 normal = glm::normalize( glm::cross(v1, v2) );

  //Store the result in global memory
  normal_map[idx] = normal;
}

extern "C" void generateNormalMap(const glm::vec3* vertex_map, glm::vec3* normal_map, const int width, const int height) {
  generateNormalMapKernel<<<width*height / 256 + 1, 256>>>(vertex_map, normal_map, width, height);
  cudaDeviceSynchronize();
}


} // namespace sensor

} // namespace octree_slam
