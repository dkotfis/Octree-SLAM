
// Octree-SLAM Dependencies
#include <octree_slam/sensor/image_kernels.h>

// CUDA / OpenGL Dependencies
#include <cuda_gl_interop.h>

namespace octree_slam {

namespace sensor {

#define PI 3.14159

int BILATERAL_KERNEL_SIZE = 7;
float BILATERAL_SIGMA_DEPTH = 40.0f; //in mm
float BILATERAL_SIGMA_SPATIAL = 4.5f;

float3 INTENSITY_RATIO = { 0.299f, 0.587f, 0.114f }; //These are taken from Kintinuous

__global__ void generateVertexMapKernel(const uint16_t* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length, const int2 img_size) {
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
  vertex_map[idx].x = ((img_size.x/width)*x - img_size.x/2) * (float) depth / focal_length.x * milli;
  vertex_map[idx].y = (img_size.y/2 - (img_size.y/height)*y) * (float) depth / focal_length.y * milli;
  vertex_map[idx].z = depth*milli;

}

extern "C" void generateVertexMap(const uint16_t* depth_pixels, glm::vec3* vertex_map, const int width, const int height, const glm::vec2 focal_length, const int2 img_size) {
  generateVertexMapKernel<<<ceil((float)width * (float)height / 256.0f), 256>>>(depth_pixels, vertex_map, width, height, focal_length, img_size);
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

  //Don't do anything for the edges, fill in invalid normals
  if (x == (width-1) || y == (height-1)) {
    normal_map[idx] = glm::vec3(INFINITY, INFINITY, INFINITY);
    return;
  }

  //Get the center point from global memory once
  glm::vec3 center = vertex_map[idx];

  //Compute two vectors within the surface (locally)
  glm::vec3 v1 = vertex_map[idx + 1] - center;
  glm::vec3 v2 = vertex_map[idx + width] - center;

  //Compute the normal
  glm::vec3 normal = glm::normalize(-glm::cross(v1, v2));

  //Store the result in global memory
  normal_map[idx] = normal;
}

extern "C" void generateNormalMap(const glm::vec3* vertex_map, glm::vec3* normal_map, const int width, const int height) {
  generateNormalMapKernel<<<ceil((float)width * (float)height / 256.0f), 256>>>(vertex_map, normal_map, width, height);
  cudaDeviceSynchronize();
}

//New and improved bilateral filter based on kinfu_remake
__global__ void bilateralKernel(const uint16_t* depth_in, uint16_t* filtered_out, const uint2 dims, const int kernel_size, const float sig_spat, const float sig_dep) {
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= dims.x*dims.y) {
    return;
  }

  int x = idx % dims.x;
  int y = idx / dims.x;

  int value = depth_in[y*dims.x + x];

  int tx = min(x - kernel_size / 2 + kernel_size, dims.x - 1);
  int ty = min(y - kernel_size / 2 + kernel_size, dims.y - 1);

  float sum1 = 0;
  float sum2 = 0;

  for (int cy = max(y - kernel_size / 2, 0); cy < ty; ++cy)
  {
    for (int cx = max(x - kernel_size / 2, 0); cx < tx; ++cx)
    {
      int depth = depth_in[cy*dims.x + cx];

      float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      float color2 = (value - depth) * (value - depth);

      float weight = __expf(-(space2 * sig_spat + color2 * sig_dep));

      sum1 += depth * weight;
      sum2 += weight;
    }
  }
  filtered_out[y*dims.x + x] = __float2int_rn(sum1 / sum2);
}

extern "C" void bilateralFilter(const uint16_t* depth_in, uint16_t* filtered_out, const int width, const int height) {
  //Use the bilateral filter kernel on the inputs
  uint2 dims = make_uint2(width, height);
  float spatial = 0.5f / (BILATERAL_SIGMA_SPATIAL * BILATERAL_SIGMA_SPATIAL);
  float depth = 0.5 / (BILATERAL_SIGMA_DEPTH * BILATERAL_SIGMA_DEPTH);
  bilateralKernel<<<ceil((float)width * (float)height/256.0f), 256>>>(depth_in, filtered_out, dims, BILATERAL_KERNEL_SIZE, spatial, depth);
  cudaDeviceSynchronize();
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
  colorToIntensityKernel<<<ceil((float)size/256.0f), 256>>>(color_in, intensity_out, size, INTENSITY_RATIO);
  cudaDeviceSynchronize();
}


__global__ void transformVertexMapKernel(glm::vec3* vertex, const glm::mat4 trans, const int size) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= size) {
    return;
  }

  vertex[idx] = glm::vec3(trans * glm::vec4(vertex[idx], 1.0f));
}

extern "C" void transformVertexMap(glm::vec3* vertex_map, const glm::mat4 &trans, const int size) {
  transformVertexMapKernel<<<ceil((float)size / 256.0f), 256>>>(vertex_map, trans, size);
}

__global__ void transformNormalMapKernel(glm::vec3* normal, const glm::mat4 trans, const int size) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= size) {
    return;
  }

  normal[idx] = glm::vec3(trans * glm::vec4(normal[idx], 0.0f));
}

extern "C" void transformNormalMap(glm::vec3* normal_map, const glm::mat4 &trans, const int size) {
  transformNormalMapKernel<<<ceil((float)size / 256.0f), 256>>>(normal_map, trans, size);
}

template <class T>
__global__ void subsampleDepthKernel(const T* data_in, T* data_out, const int width, const int height, const float sigma_depth) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= width*height) {
    return;
  }

  //Compute the x/y coords of this thread
  int x = idx % width;
  int y = idx / width;

  const int D = 5;
  float center = data_in[4*y*width + 2*x];

  int tx = min(2 * x - D / 2 + D, 2*width - 1);
  int ty = min(2 * y - D / 2 + D, 2*height - 1);

  float sum = 0;
  float count = 0;

  for (int cy = max(0, 2 * y - D / 2); cy < ty; ++cy) {
    for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
      float val = data_in[2*cy*width + cx];
      if (abs(val - center) < sigma_depth) {
        sum += val;
        ++count;
      }
    }
  }

  data_out[y*width + x] = (T) (count == 0) ? 0 : sum / count;
}

template <class T>
void subsampleDepth(T* data, const int width, const int height) {
  //Create new memory space (this can't actually be done in place)
  T* data_new;
  cudaMalloc((void**)&data_new, width*height*sizeof(T)/4);

  subsampleDepthKernel<<<ceil((float)width * (float)height/1024.0f), 256>>>(data, data_new, width/2, height/2, BILATERAL_SIGMA_DEPTH*3.0f);
  cudaDeviceSynchronize();

  //Copy into the input
  cudaMemcpy(data, data_new, width*height*sizeof(T)/4, cudaMemcpyDeviceToDevice);

  //Free the temporary memory slot
  cudaFree(data_new);
}

//Declare types to generate symbols
template void subsampleDepth<uint16_t>(uint16_t* data, const int width, const int height);
template void subsampleDepth<float>(float* data, const int width, const int height);

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
  data_out[y*width + x] = data_in[4 * y*width + 2 * x];
}

template <class T>
void subsample(T* data, const int width, const int height) {
  //Create new memory space (this can't actually be done in place)
  T* data_new;
  cudaMalloc((void**)&data_new, width*height*sizeof(T) / 4);

  subsampleKernel << <width*height / 1024 + 1, 256 >> >(data, data_new, width / 2, height / 2);
  cudaDeviceSynchronize();

  //Copy into the input
  cudaMemcpy(data, data_new, width*height*sizeof(T) / 4, cudaMemcpyDeviceToDevice);

  //Free the temporary memory slot
  cudaFree(data_new);
}

//Declare types to generate symbols
template void subsample<Color256>(Color256* data, const int width, const int height);
template void subsample<float>(float* data, const int width, const int height);

} // namespace sensor

} // namespace octree_slam
