#ifndef IMAGE_KERNELS_H_
#define IMAGE_KERNELS_H_

// CUDA Dependencies
#include <cuda_runtime.h>

// OpenGL Dependency
#include <glm/glm.hpp>

// OpenNI Dependency
#include <OpenNI.h> //TODO: Why is this needed?

// Octree-SLAM Dependency
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace sensor {

//Generates a vertex map from a depth image
extern "C" void generateVertexMap(const uint16_t* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length);

//Generates a normal map from a vertex map
extern "C" void generateNormalMap(const glm::vec3* vertex_map, glm::vec3* normal_map, const int width, const int height);

//Applies a bilateral filter to a depth image
extern "C" void bilateralFilter(const uint16_t* depth_in, uint16_t* filtered_out, const int width, const int height);

//Applies a gaussian filter (in place)
template <class T>
void gaussianFilter(T* data, const int width, const int height);

//Subsamples (in place) to construct a pyramid
template <class T>
void subsample(T* data, const int width, const int height);

//Computes an intensity image from a color input
extern "C" void colorToIntensity(const Color256* color_in, float* intensity_out, const int size);

//Takes a gradient of an intensity image using a sobel filter
extern "C" void gradient(const float* intensity_in, glm::vec2* gradient, const int width, const int height);

//Computes a difference between two intensity images
extern "C" void difference(const float* in1, const float* in2, float* out, const int size);

//Transforms a vertex map (in place) with an input transformation matrix
extern "C" void transformVertexMap(glm::vec3* vertex_map, const glm::mat4 &trans, const int size);

//Transforms a normal map (in place) with an input transformation matrix
extern "C" void transformNormalMap(glm::vec3* normal_map, const glm::mat4 &trans, const int size);

} // namespace sensor

} // namespace octree_slam

#endif //IMAGE_KERNELS_H_
