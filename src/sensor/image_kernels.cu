
// Octree-SLAM Dependencies
#include <octree_slam/sensor/image_kernels.h>

// CUDA / OpenGL Dependencies
#include <cuda_gl_interop.h>

namespace octree_slam {

namespace sensor {

#define PI 3.14159

int GAUSS_RADIUS = 2;
float GAUSS_SIGMA = 100.0;
int BILATERAL_RADIUS = 2;
float BILATERAL_SIGMA = 100.0;
float3 INTENSITY_RATIO = { 0.299f, 0.587f, 0.114f }; //These are taken from Kintinuous

__global__ void generateVertexMapKernel(const uint16_t* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length) {
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

extern "C" void generateVertexMap(const uint16_t* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length) {
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

__host__ __device__ float gaussian1d(float x, float sigma) {
  float variance = pow(sigma, 2);
  float exponent = -pow(x, 2) / (2 * variance);
  return expf(exponent) / sqrt(2 * PI * variance);
}

__host__ __device__ uint2 idx_to_co(unsigned int idx, uint2 dims) {
  uint2 res;
  res.x = idx % dims.x;
  res.y = idx / dims.x;
  return res;
}

__host__ __device__ unsigned int co_to_idx(uint2 co, uint2 dims) {
  unsigned int res;
  res = co.y * dims.x + co.x;
  return res;
}

//This is borrowed from http://cs.au.dk/~staal/dpc/20072300_paper_final.pdf
__global__ void bilateralFilterGPU_v2(const uint16_t* input, uint16_t* output, uint2 dims, int radius, float* kernel, float sigma_range) {
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= dims.x*dims.y) {
    return;
  }

  uint2 pos = idx_to_co(idx, dims);
  float currentColor = input[idx];
  float res = 0;
  float normalization = 0;

  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      int x_sample = pos.x + i;
      int y_sample = pos.y + j;

      //mirror edges
      if (x_sample < 0) x_sample = -x_sample;
      if (y_sample < 0) y_sample = -y_sample;
      if (x_sample > dims.x - 1) x_sample = dims.x - 1 - i;
      if (y_sample > dims.y - 1) y_sample = dims.y - 1 - j;
      float tmpColor =
        input[co_to_idx(make_uint2(x_sample, y_sample), dims)];

      //Don't continue if its a bad pixel
      if (tmpColor == 0 || tmpColor == 65535) {
        continue;
      }
      float gauss_spatial =
        kernel[co_to_idx(make_uint2(i + radius, j + radius), make_uint2(radius *
        2 + 1, radius * 2 + 1))];
      float gauss_range;
      gauss_range = gaussian1d(currentColor - tmpColor,
        sigma_range);
      float weight = gauss_spatial*gauss_range;
      normalization = normalization + weight;
      res = res + (tmpColor * weight);
    }
  }
  res /= normalization;

  output[idx] = res;
}

//This is borrowed and is used to compute a gaussian function on CPU or GPU
__host__ __device__ float gaussian2d(float x, float y, float sigma) {
  float variance = pow(sigma, 2);
  float exponent = -(pow(x, 2) + pow(y, 2)) / (2 * variance);
  return expf(exponent) / (2 * PI * variance);
}

//This is similarly borrowed to precompute the gaussian on the CPU
float* generateGaussianKernel(int radius, float sigma) {
  int area = (2 * radius + 1)*(2 * radius + 1);
  float* res = new float[area];
  for (int x = -radius; x <= radius; x++)
    for (int y = -radius; y <= radius; y++)
    {
    //Co_to_idx inspired
    int position = (x + radius)*(radius * 2 + 1) + y + radius;
    res[position] = gaussian2d(x, y, sigma);
    }
  return res;
}

/*
//TODO: Adapt to using texture memory so this more efficient implementation can be used
__global__ void bilateralFilterGPU_v5(float3* output, uint2 dims, int radius, float* kernel, float variance, float sqrt_pi_variance) {
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint2 pos = idx_to_co(idx, dims);
  if (pos.x >= dims.x || pos.y >= dims.y) return;
  float3 currentColor = make_float3(tex1Dfetch(tex,
    3 * idx), tex1Dfetch(tex, 3 * idx + 1), tex1Dfetch(tex, 3 * idx + 2));
  float3 res = make_float3(0.0f, 0.0f, 0.0f);
  float3 normalization = make_float3(0.0f, 0.0f, 0.0f);
  float3 weight;
  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      int x_sample = pos.x + i;
      int y_sample = pos.y + j;
      //mirror edges
      if (x_sample < 0) x_sample = -x_sample;
      if (y_sample < 0) y_sample = -y_sample;
      if (x_sample > dims.x - 1) x_sample = dims.x - 1 - i;
      if (y_sample > dims.y - 1) y_sample = dims.y - 1 - j;
      int tempPos =
        co_to_idx(make_uint2(x_sample, y_sample), dims);
      float3 tmpColor = make_float3(tex1Dfetch(tex,
        3 * tempPos), tex1Dfetch(tex, 3 * tempPos + 1), tex1Dfetch(tex,
        3 * tempPos + 2));//input[tempPos];
      float gauss_spatial =
        kernel[co_to_idx(make_uint2(i + radius, j + radius), make_uint2(radius *
        2 + 1, radius * 2 + 1))];
      weight.x = gauss_spatial *
        gaussian1d_gpu_reg((currentColor.x -
        tmpColor.x), variance, sqrt_pi_variance);
      weight.y = gauss_spatial *
        gaussian1d_gpu_reg((currentColor.y -
        tmpColor.y), variance, sqrt_pi_variance);
      weight.z = gauss_spatial *
        gaussian1d_gpu_reg((currentColor.z -
        tmpColor.z), variance, sqrt_pi_variance);
      normalization = normalization + weight;
      res = res + (tmpColor * weight);
    }
  }
  res.x /= normalization.x;
  res.y /= normalization.y;
  res.z /= normalization.z;
  output[idx] = res;
}
*/

extern "C" void bilateralFilter(const uint16_t* depth_in, uint16_t* filtered_out, const int width, const int height) {
  //Create the gaussian kernel and transfer to GPU memory
  float* kernel = generateGaussianKernel(BILATERAL_RADIUS, 10.0);
  float* d_kernel;
  cudaMalloc((void**)&d_kernel, (BILATERAL_RADIUS * 2 + 1)*(BILATERAL_RADIUS * 2 + 1)*sizeof(float));
  cudaMemcpy(d_kernel, kernel, (BILATERAL_RADIUS * 2 + 1)*(BILATERAL_RADIUS * 2 + 1)*sizeof(float), cudaMemcpyHostToDevice);
  delete kernel;

  //Use the bilateral filter kernel on the inputs
  uint2 dims = make_uint2(width, height);
  bilateralFilterGPU_v2<<<width*height/256 + 1, 256>>>(depth_in, filtered_out, dims, BILATERAL_RADIUS, d_kernel, BILATERAL_SIGMA);
  cudaDeviceSynchronize();
  cudaFree(d_kernel);
}

__global__ void colorToIntensityKernel(const Color256* color_in, float* intensity_out, const int size, const float3 intensity_ratio) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= size) {
    return;
  }

  intensity_out[idx] = color_in[idx].r/255.0f * intensity_ratio.x + color_in[idx].b/255.0f * intensity_ratio.y 
    + color_in[idx].b/255.0f * intensity_ratio.z;
}

extern "C" void colorToIntensity(const Color256* color_in, float* intensity_out, const int size) {
  colorToIntensityKernel<<<size/256 + 1, 256>>>(color_in, intensity_out, size, INTENSITY_RATIO);
  cudaDeviceSynchronize();
}


__global__ void transformVertexMapKernel(glm::vec3* vertex, const glm::mat4 trans, const int size, const int load_size) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx*load_size >= size) {
    return;
  }

  //Determine whether the full load is in the bounds
  int bound = load_size;
  if ((idx+1)*load_size - size > 0 ) {
    bound -= (idx + 1)*load_size - size;
  }

  for (size_t i = 0; i < bound; i++) {
    vertex[load_size*idx + i] = glm::vec3(trans*glm::vec4(vertex[load_size*idx + i], 1.0f));
  }
}

extern "C" void transformVertexMap(glm::vec3* vertex_map, const glm::mat4 &trans, const int size) {
  int load_size = 16;
  transformVertexMapKernel<<<size / 256 / load_size + 1, 256>>>(vertex_map, trans, size, load_size);
}

__global__ void transformNormalMapKernel(glm::vec3* normal, const glm::mat4 trans, const int size, const int load_size) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx*load_size >= size) {
    return;
  }

  //Determine whether the full load is in the bounds
  int bound = load_size;
  if ((idx + 1)*load_size - size > 0) {
    bound -= (idx + 1)*load_size - size;
  }

  for (size_t i = 0; i < bound; i++) {
    normal[load_size*idx + i] = glm::vec3(trans*glm::vec4(normal[load_size*idx + i], 0.0f));
  }
}

extern "C" void transformNormalMap(glm::vec3* normal_map, const glm::mat4 &trans, const int size) {
  int load_size = 16;
  transformNormalMapKernel<<<size / 256 / load_size + 1, 256>>>(normal_map, trans, size, load_size);
}

template <class T>
__global__ void gaussianFilterKernel(const T* input, T* output, uint2 dims, int radius, float* kernel) {
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint2 pos = idx_to_co(idx, dims);
  int img_x = pos.x;
  int img_y = pos.y;
  if (img_x >= dims.x || img_y >= dims.y) return;
  float res = 0;
  float normalization = 0;
  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      int x_sample = img_x + i;
      int y_sample = img_y + j;
      //mirror edges
      if (x_sample < 0) x_sample = -x_sample;
      if (y_sample < 0) y_sample = -y_sample;
      if (x_sample > dims.x - 1) x_sample = dims.x - 1 - i;
      if (y_sample > dims.y - 1) y_sample = dims.y - 1 - j;
      float tmpColor =
        input[co_to_idx(make_uint2(x_sample, y_sample), dims)];
      float gauss_spatial =
        kernel[co_to_idx(make_uint2(i + radius, j + radius), make_uint2(radius *
        2 + 1, radius * 2 + 1))];
      normalization = normalization + gauss_spatial;
      res = res + (tmpColor * gauss_spatial);
    }
  }
  res /= normalization;
  output[idx] = res;
}

template <class T>
void gaussianFilter(T* data, const int width, const int height) {
  //Create the gaussian kernel and transfer to GPU memory
  float* kernel = generateGaussianKernel(GAUSS_RADIUS, GAUSS_SIGMA);
  float* d_kernel;
  cudaMalloc((void**)&d_kernel, (2*GAUSS_RADIUS+1)*(2*GAUSS_RADIUS+1)*sizeof(float));
  cudaMemcpy(d_kernel, kernel, (2*GAUSS_RADIUS+1)*(2*GAUSS_RADIUS+1)*sizeof(float), cudaMemcpyHostToDevice);
  delete kernel;

  //Create new memory space (this can't actually be done in place)
  T* data_new;
  cudaMalloc((void**)&data_new, width*height*sizeof(T));

  //Use the bilateral filter kernel on the inputs
  uint2 dims = make_uint2(width, height);
  gaussianFilterKernel<<<width*height / 256 + 1, 256>>>(data, data_new, dims, GAUSS_RADIUS, d_kernel);
  cudaDeviceSynchronize();

  //Copy into the input
  cudaMemcpy(data, data_new, width*height*sizeof(T), cudaMemcpyDeviceToDevice);

  //Free the temporary memory slot
  cudaFree(data_new);
  cudaFree(d_kernel);
}

//template void gaussianFilter<Color256>(Color256* data, const int width, const int height);
template void gaussianFilter<uint16_t>(uint16_t* data, const int width, const int height);
template void gaussianFilter<float>(float* data, const int width, const int height);

template <class T>
__global__ void subsampleKernel(const T* data_in, T* data_out, const int width, const int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= width*height) {
    return;
  }

  //Compute the x/y coords of this thread
  int x = idx % width;
  int y = idx / width;

  //Sample the value
  data_out[y*width + x] = data_in[4*y*width + 2*x];
}

template <class T>
void subsample(T* data, const int width, const int height) {
  //Create new memory space (this can't actually be done in place)
  T* data_new;
  cudaMalloc((void**)&data_new, width*height*sizeof(T));

  subsampleKernel<<<width*height/1024 + 1, 256>>>(data, data_new, width/2, height/2);
  cudaDeviceSynchronize();

  //Copy into the input
  cudaMemcpy(data, data_new, width*height*sizeof(T), cudaMemcpyDeviceToDevice);

  //Free the temporary memory slot
  cudaFree(data_new);
}

//Declare types to generate symbols
template void subsample<Color256>(Color256* data, const int width, const int height);
template void subsample<uint16_t>(uint16_t* data, const int width, const int height);
template void subsample<float>(float* data, const int width, const int height);

} // namespace sensor

} // namespace octree_slam
