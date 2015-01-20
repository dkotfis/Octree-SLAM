#ifndef CUDA_COMMON_KERNELS_H_
#define CUDA_COMMON_KERNELS_H_

//Handy function for clamping between two values;
__host__ __device__ inline float clamp(float val, float min, float max) {
  if (val < min) {
    val = min;
  }
  else if (val > max) {
    val = max;
  }
  return val;
}

#endif //CUDA_COMMON_KERNELS_H_
