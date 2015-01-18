#ifndef OPENNI_KERNELS_H_
#define OPENNI_KERNELS_H_

// CUDA Dependencies
#include <cuda.h>

// OpenGL Dependency
#include <glm/glm.hpp>

// OpenNI Dependency
#include <OpenNI.h>

// Forward Declarations
class uchar4;

namespace octree_slam {

namespace sensor {

//Helper function for CUDA Kernel that converts a color frame to PBO format for rendering
extern "C" void writeColorToPBO(const openni::RGB888Pixel* color_pixels, uchar4* pbo, const int num_pixels);

//Helper function for CUDA Kernel that generates a vertex map from a depth image
extern "C" void generateVertexMap(const openni::DepthPixel* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length);

//Helper function for CUDA Kernel that generates a normal map from a vertex map
extern "C" void generateNormalMap(const glm::vec3* vertex_map, glm::vec3* normal_map, const int width, const int height);

} // namespace sensor

} // namespace octree_slam

#endif //OPENNI_KERNELS_H_
