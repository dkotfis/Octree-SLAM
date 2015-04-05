
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cuda.h>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>

// GL Dependency
#include <glm/gtc/matrix_transform.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/rendering/cone_tracing_kernels.h>

namespace octree_slam {

namespace rendering {

//The maximum distance that we can see
const float MAX_RANGE = 10.0f;

//The starting distance from the origin to start the ray marching from
const float START_DIST = 0.2f;

__global__ void createRays(glm::vec2 resolution, float fov, glm::vec3 x_dir, glm::vec3 y_dir, glm::vec3* rays) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= resolution.x*resolution.y) {
    return;
  }

  //Compute the x/y coords of this thread
  int x = idx % (int)resolution.x;
  int y = idx / (int)resolution.x;

  //Calculate the perpindicular vector component magnitudes from the fov, resolution, and x/y values
  float fac = tan(fov) / resolution.y;
  glm::vec2 mag;
  mag.x = fac * ( (float)x - resolution.x / 2.0f );
  mag.y = fac * ( (float)y - resolution.y / 2.0f );

  //Calculate the direction
  rays[idx] = (mag.x * x_dir) + (mag.y * y_dir);

}

__global__ void coneTrace(uchar4* pos, int* ind, int numRays, glm::vec3 camera_origin, glm::vec3* rays, float distance, int depth, unsigned int* octree, glm::vec3 oct_center, float oct_size) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= numRays) {
    return;
  }

  int index = ind[idx];

  //Compute the target point
  glm::vec3 target = camera_origin + rays[index]*distance;

  //Descend into the octree and get the value
  unsigned int oct_val;
  int node_idx = 0;
  int child_idx = 0;
  bool is_occupied = true;
  for (int i = 0; i < depth; i++) {
    //Determine which octant the point lies in
    bool x = target.x > oct_center.x;
    bool y = target.y > oct_center.y;
    bool z = target.z > oct_center.z;

    //Update the child number
    int child = (x + 2 * y + 4 * z);

    //Get the child number from the first three bits of the morton code
    node_idx = child_idx + child;

    if (!octree[2 * node_idx] & 0x40000000) {
      is_occupied = false;
      break;
    }

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;

    //Update the edge length
    oct_size /= 2.0f;

    //Update the center
    oct_center.x += oct_size * (x ? 1 : -1);
    oct_center.y += oct_size * (y ? 1 : -1);
    oct_center.z += oct_size * (z ? 1 : -1);
  }

  //Update the pixel value
  if (is_occupied) {
    uchar4 value = pos[index];
    int alpha = oct_val >> 24;
    value.x = (alpha/255.0f)*((oct_val & 0xFF));
    value.y = (alpha/255.0f)*((oct_val >> 8) & 0xFF);
    value.z = (alpha/255.0f)*((oct_val >> 16) & 0xFF);
    pos[index] = value;

    //Flag the ray as finished if alpha is saturated
    if ((int)value.w + alpha > 255) {
      value.w += alpha;
    } else {
      value.w = 255;
      //TODO: Flag here
    }
  }

}

struct is_negative
{
  __host__ __device__
    bool operator()(const int &x)
  {
    return x < 0;
  }
};

extern "C" void coneTraceSVO(uchar4* pos, glm::vec2 resolution, float fov, glm::mat4 cameraPose, SVO octree) {

  int numRays = (int)resolution.x * (int)resolution.y;

  glm::vec3 camera_origin = glm::vec3(glm::inverse(cameraPose)*glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

  //Create rays
  glm::vec3* rays;
  cudaMalloc((void**)&rays, numRays * sizeof(glm::vec3));
  glm::vec3 x_dir = glm::vec3(glm::inverse(cameraPose) * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
  glm::vec3 y_dir = glm::vec3(glm::inverse(cameraPose) * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
  createRays<<<ceil(numRays / 256.0f), 256>>>(resolution, fov, x_dir, y_dir, rays);

  //Initialize distance and depth
  float distance = START_DIST;
  float pix_size = distance*tan(fov*3.14159f/180.0f)/resolution.y;
  int depth = ceil(log((float)(octree.size/pix_size)) / log(2.0f));

  //Setup indices
  int* ind;
  cudaMalloc((void**)&ind, numRays*sizeof(glm::vec3));
  thrust::device_ptr<int> t_ind = thrust::device_pointer_cast<int>(ind);
  thrust::sequence(t_ind, t_ind+numRays, 0, 1);

  //Loop Cone trace kernel
  while (numRays > 0 && distance < MAX_RANGE) {
    //Call the cone tracer
    coneTrace<<<ceil(numRays / 256.0f), 256>>>(pos, ind, numRays, camera_origin, rays, distance, depth, octree.data, octree.center, octree.size);

    //Use thrust to remove rays that are saturated
    numRays = thrust::remove_if(t_ind, t_ind + numRays, is_negative()) - t_ind;

    //Update the distance and depth
    distance += octree.size / (float) pow(2, depth);
    pix_size = distance*tan(fov*3.14159f / 180.0f) / resolution.y;
    depth = ceil(log((float)(octree.size / pix_size)) / log(2.0f));
  }

  //Cleanup
  cudaFree(rays);
  cudaFree(ind);
}

} // namespace rendering

} // namespace octree_slam
