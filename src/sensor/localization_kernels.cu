
// CUDA Dependencies
#include <cuda.h>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

// Octree-SLAM Dependencies
#include <octree_slam/sensor/localization_kernels.h>

namespace octree_slam {

namespace sensor {

__device__ const float DIST_THRESH = 0.10f; //Use 10 cm distance threshold for correspondences
__device__ const float NORM_THRESH = 0.7f; //Use 30% orientation threshold for correspondences

//Define structures to be used for Mat6x6 and Vec6 for thrust summation
struct Mat6x6 {
  float values[36];
  __host__ __device__ Mat6x6() {};
  __host__ __device__ Mat6x6(const int val) {
    for (int i = 0; i < 36; i++) {
      values[i] = val;
    }
  };
};

__host__ __device__ inline Mat6x6 operator+(const Mat6x6& lhs, const Mat6x6& rhs) {
  Mat6x6 result;
  for (int i = 0; i < 36; i++) {
    result.values[i] = lhs.values[i] + rhs.values[i];
  }
  return result;
}

struct Vec6 {
  float values[6];
  __host__ __device__ Vec6() {};
  __host__ __device__ Vec6(const int val) {
    for (int i = 0; i < 6; i++) {
      values[i] = val;
    }
  };
};

__host__ __device__ inline Vec6 operator+(const Vec6& lhs, const Vec6& rhs) {
  Vec6 result;
  for (int i = 0; i < 6; i++) {
    result.values[i] = lhs.values[i] + rhs.values[i];
  }
  return result;
}

ICPFrame::ICPFrame(const int w, const int h) : width(w), height(h) {
  cudaMalloc((void**)&vertex, width*height*sizeof(glm::vec3));
  cudaMalloc((void**)&normal, width*height*sizeof(glm::vec3));
};

ICPFrame::~ICPFrame() {
  cudaFree(vertex);
  cudaFree(normal);
}

RGBDFrame::RGBDFrame(const int w, const int h) : width(w), height(h) {
  cudaMalloc((void**)&intensity, width*height*sizeof(float));
  cudaMalloc((void**)&vertex, width*height*sizeof(glm::vec3));
}

RGBDFrame::~RGBDFrame() {
  cudaFree(intensity);
  cudaFree(vertex);
}

__global__ void computeICPCorrespondences(const glm::vec3* last_frame_vertex, const glm::vec3* last_frame_normal, const glm::vec3* this_frame_vertex, const glm::vec3* this_frame_normal, 
    const int num_points, bool* stencil, int* num_corr) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= num_points) {
    return;
  }

  bool is_match = true;

  //Check whether points are any good
  if (!isfinite(this_frame_vertex[idx].x) || !isfinite(this_frame_vertex[idx].y) || !isfinite(this_frame_vertex[idx].z)
    || !isfinite(last_frame_vertex[idx].x) || !isfinite(last_frame_vertex[idx].y) || !isfinite(last_frame_vertex[idx].z)) {
    is_match = false;
  }
  if (!is_match || !isfinite(this_frame_normal[idx].x) || !isfinite(this_frame_normal[idx].y) || !isfinite(this_frame_normal[idx].z)
    || !isfinite(last_frame_normal[idx].x) || !isfinite(last_frame_normal[idx].y) || !isfinite(last_frame_normal[idx].z)) {
    is_match = false;
  }

  //Check position difference
  if (!is_match || glm::length(this_frame_vertex[idx] - last_frame_vertex[idx]) > DIST_THRESH) {
    is_match = false;
  }

  //Check normal difference
  if (!is_match || glm::dot(this_frame_normal[idx], last_frame_normal[idx]) < NORM_THRESH) {
    is_match = false;
  }

  //Update result
  stencil[idx] = is_match;

  //Subtract from global counter if its not a match
  if (!is_match) {
    atomicAdd(num_corr, -1);
  }

}

__global__ void computeICPCostsKernel(const glm::vec3* last_frame_normal, const glm::vec3* last_frame_vertex, const glm::vec3* this_frame_vertex, const int num_points, Mat6x6* As, Vec6* bs) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= num_points) {
    return;
  }

  //Get the vertex and normal values
  glm::vec3 v2 = this_frame_vertex[idx];
  glm::vec3 v1 = last_frame_vertex[idx];
  glm::vec3 n = last_frame_normal[idx];

  //Construct A_T
  float G_T[18] = { 0.0f, -v2.z, v2.y, v2.z, 0.0f, -v2.x, -v2.y, v2.x, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
  float A_T[6];
  for (int i = 0; i < 6; i++) {
    A_T[i] = G_T[3 * i] * n.x + G_T[3*i + 1] * n.y + G_T[3*i + 2] * n.z;
  }

  //Construct b
  float b = glm::dot(n, v1 - v2);

  //Compute outputs
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      As[idx].values[6*i + j] = A_T[i] * A_T[j];
    }
    bs[idx].values[i] = b*A_T[i];
  }
}

__global__ void computeICPCostsUncorrespondedKernel(const glm::vec3* last_frame_normal, const glm::vec3* last_frame_vertex, const glm::vec3* this_frame_normal, 
  const glm::vec3* this_frame_vertex, const int num_points, Mat6x6* As, Vec6* bs) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= num_points) {
    return;
  }

  //Init outputs
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      As[idx].values[6 * i + j] = 0.0f;
    }
    bs[idx].values[i] = 0.0f;
  }

  //Get the vertex and normal values
  glm::vec3 v2 = this_frame_vertex[idx];
  glm::vec3 n2 = this_frame_normal[idx];
  glm::vec3 v1 = last_frame_vertex[idx];
  glm::vec3 n1 = last_frame_normal[idx];

  //Check whether points are any good
  if (!isfinite(v2.x) || !isfinite(v2.y) || !isfinite(v2.z)
    || !isfinite(v1.x) || !isfinite(v1.y) || !isfinite(v1.z)) {
    return;
  }
  if (!isfinite(n2.x) || !isfinite(n2.y) || !isfinite(n2.z)
    || !isfinite(n1.x) || !isfinite(n1.y) || !isfinite(n1.z)) {
    return;
  }

  //Check position difference
  if (glm::length(v2 - v1) > DIST_THRESH) {
    return;
  }

  //Check normal difference
  if (glm::dot(n2, n1) < NORM_THRESH) {
    return;
  }

  //Construct A_T
  float G_T[18] = { 0.0f, -v2.z, v2.y, v2.z, 0.0f, -v2.x, -v2.y, v2.x, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
  float A_T[6];
  for (int i = 0; i < 6; i++) {
    A_T[i] = G_T[3 * i] * n1.x + G_T[3 * i + 1] * n1.y + G_T[3 * i + 2] * n1.z;
  }

  //Construct b
  float b = glm::dot(n1, v1 - v2);

  //Compute outputs
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      As[idx].values[6 * i + j] = A_T[i] * A_T[j];
    }
    bs[idx].values[i] = b*A_T[i];
  }
}

extern "C" void computeICPCost(const ICPFrame* last_frame, const ICPFrame &this_frame, float* A, float* b) {
  //TODO: Verify that the two frames are the same size

  //Compute correspondences 
  int num_correspondences = this_frame.width * this_frame.height;
  bool* d_stencil;
  int* d_num_corr;
  cudaMalloc((void**)&d_stencil, num_correspondences * sizeof(bool));
  cudaMalloc((void**)&d_num_corr, sizeof(int));
  cudaMemcpy(d_num_corr, &num_correspondences, sizeof(int), cudaMemcpyHostToDevice); //Initialize to the total points. Assume that most points will be valid
  computeICPCorrespondences<<<num_correspondences / 256 + 1, 256>>>(last_frame->vertex, last_frame->normal, this_frame.vertex, this_frame.normal, 
    num_correspondences, d_stencil, d_num_corr);
  cudaDeviceSynchronize();

  //Copy number of correspondences back from the device
  cudaMemcpy(&num_correspondences, d_num_corr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_num_corr);

  //Don't continue without any correspondences
  if (num_correspondences <= 0) {
    return;
  }

  //Allocate memory for reduced copies
  glm::vec3* last_frame_reduced_vertex;
  cudaMalloc((void**)&last_frame_reduced_vertex, num_correspondences * sizeof(glm::vec3));
  glm::vec3* last_frame_reduced_normal;
  cudaMalloc((void**)&last_frame_reduced_normal, num_correspondences * sizeof(glm::vec3));
  glm::vec3* this_frame_reduced_vertex;
  cudaMalloc((void**)&this_frame_reduced_vertex, num_correspondences * sizeof(glm::vec3));

  //Reduce inputs with thrust compaction
  thrust::device_ptr<glm::vec3> in, out;
  thrust::device_ptr<bool> sten = thrust::device_pointer_cast<bool>(d_stencil);
  in = thrust::device_pointer_cast<glm::vec3>(last_frame->vertex);
  out = thrust::device_pointer_cast<glm::vec3>(last_frame_reduced_vertex);
  thrust::copy_if(in, in + last_frame->width*last_frame->height, sten, out, thrust::identity<bool>());
  in = thrust::device_pointer_cast<glm::vec3>(last_frame->normal);
  out = thrust::device_pointer_cast<glm::vec3>(last_frame_reduced_normal);
  thrust::copy_if(in, in + last_frame->width*last_frame->height, sten, out, thrust::identity<bool>());
  in = thrust::device_pointer_cast<glm::vec3>(this_frame.vertex);
  out = thrust::device_pointer_cast<glm::vec3>(this_frame_reduced_vertex);
  thrust::copy_if(in, in + last_frame->width*last_frame->height, sten, out, thrust::identity<bool>());
  
  //Free device memory from data in the compaction stages
  cudaFree(d_stencil);

  //Compute cost terms
  Mat6x6* d_A;
  Vec6* d_b;
  cudaMalloc((void**) &d_A, num_correspondences * sizeof(Mat6x6));
  cudaMalloc((void**) &d_b, num_correspondences * sizeof(Vec6));
  computeICPCostsKernel<<<num_correspondences / 256 + 1, 256>>>(last_frame_reduced_normal, last_frame_reduced_vertex, this_frame_reduced_vertex, num_correspondences, d_A, d_b);
  cudaDeviceSynchronize();

  //Free up device memory
  cudaFree(last_frame_reduced_vertex);
  cudaFree(last_frame_reduced_normal);
  cudaFree(this_frame_reduced_vertex);

  //Sum terms (reduce) with thrust
  thrust::device_ptr<Mat6x6> thrust_A = thrust::device_pointer_cast<Mat6x6>(d_A);
  Mat6x6 matA = thrust::reduce(thrust_A, thrust_A + num_correspondences);
  thrust::device_ptr<Vec6> thrust_b = thrust::device_pointer_cast<Vec6>(d_b);
  Vec6 vecb = thrust::reduce(thrust_b, thrust_b + num_correspondences);

  //Free up device memory
  cudaFree(d_A);
  cudaFree(d_b);

  //Copy result to output
  memcpy(A, matA.values, 36 * sizeof(float));
  memcpy(b, vecb.values, 6 * sizeof(float));
}

extern "C" void computeICPCost2(const ICPFrame* last_frame, const ICPFrame &this_frame, float* A, float* b) {
  //TODO: Verify that the two frames are the same size

  //Assume all are correspondences 
  int num_correspondences = this_frame.width * this_frame.height;

  //Compute cost terms
  Mat6x6* d_A;
  Vec6* d_b;
  cudaMalloc((void**)&d_A, num_correspondences * sizeof(Mat6x6));
  cudaMalloc((void**)&d_b, num_correspondences * sizeof(Vec6));
  computeICPCostsUncorrespondedKernel << <num_correspondences / 256 + 1, 256 >> >(last_frame->normal, last_frame->vertex, this_frame.normal, this_frame.vertex, num_correspondences, d_A, d_b);
  cudaDeviceSynchronize();

  //Sum terms (reduce) with thrust
  thrust::device_ptr<Mat6x6> thrust_A = thrust::device_pointer_cast<Mat6x6>(d_A);
  Mat6x6 matA = thrust::reduce(thrust_A, thrust_A + num_correspondences);
  thrust::device_ptr<Vec6> thrust_b = thrust::device_pointer_cast<Vec6>(d_b);
  Vec6 vecb = thrust::reduce(thrust_b, thrust_b + num_correspondences);

  //Free up device memory
  cudaFree(d_A);
  cudaFree(d_b);

  //Copy result to output
  memcpy(A, matA.values, 36 * sizeof(float));
  memcpy(b, vecb.values, 6 * sizeof(float));
}

extern "C" void computeRGBDCost(const RGBDFrame* last_frame, const RGBDFrame& this_frame, float* A, float* b) {
  //TODO: Stuff here
  cudaDeviceSynchronize();
}

} // namespace sensor

} // namespace octree_slam
