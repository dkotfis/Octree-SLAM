#ifndef CONE_TRACING_KERNELS_H_
#define CONE_TRACING_KERNELS_H_

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

#include <octree_slam/utilities.h>
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace rendering {

extern "C" void coneTraceSVO(uchar4* pos, glm::vec2 resolution, float fov, glm::mat4 cameraPose, SVO octree);

} // namespace rendering

} // namespace octree_slam

#endif //CONE_TRACING_KERNELS_H_
