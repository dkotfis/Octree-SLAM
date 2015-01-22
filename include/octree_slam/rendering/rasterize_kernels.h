#ifndef RASTERIZE_KERNEL_H_
#define RASTERIZE_KERNEL_H_

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

#include <octree_slam/utilities.h>
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace rendering {

void kernelCleanup();
extern "C" void rasterizeMesh(uchar4* pos, glm::vec2 resolution, glm::mat4 rotationM,float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, const bmp_texture *tex, std::vector<glm::vec4> *texcoord, glm::mat4 view, glm::vec3 lightpos, int mode, bool barycenter);

//Helper function for CUDA Kernel that converts a color frame to PBO format for rendering
extern "C" void writeColorToPBO(const Color256* color_pixels, uchar4* pbo, const int num_pixels);

} // namespace rendering

} // namespace octree_slam

#endif //RASTERIZE_KERNEL_H_
