
#include <vector>

// Octree-SLAM Dependencies
#include <octree_slam/sensor/rgbd_camera.h>
#include <octree_slam/sensor/localization_kernels.h>

// OpenGL Dependency
#include <glm/gtc/matrix_transform.hpp>

// CUDA Dependencies
#include <cuda_runtime.h>

namespace octree_slam {

namespace sensor {

RGBDCamera::RGBDCamera(const int width, const int height, const glm::vec2 &focal_length) : 
  width_(width), height_(height), focal_length_(focal_length), has_frame_(false) {
}

RGBDCamera::~RGBDCamera() {

}

const Camera RGBDCamera::camera() const {
  Camera cam;

  cam.model = glm::mat4(1.0f);
  cam.view = glm::mat4(orientation_) * glm::translate(glm::mat4(1.0f), position_);
  //cam.projection = TODO 
  cam.modelview = cam.view * cam.model;
  cam.mvp = cam.projection*cam.modelview;

  return cam;
}

void RGBDCamera::update(const Frame& this_frame) {
  if (has_frame_) {
    //TODO: Create pyramids and loop over them

    //Get the ICP cost values
    float A[6*6];
    float b[6];
    computeICPCost(last_frame_, this_frame, A, b);
    //TODO: Use RBGD cost also, and create weighted sum as in Kintinuous

    //Solve for the optimized camera transformation
    float x[6];
    solveCholesky(6, A, b, x);

    //Update position/orientation of the camera
    glm::mat4 update_trans = glm::mat4(glm::mat3(1.0f, x[2], -x[1], -x[2], 1.0f, x[0], x[1], -x[0], 1.0f)) 
      * glm::translate(glm::mat4(1.0f), glm::vec3(x[3], x[4], x[5]));
    position_ = glm::vec3( update_trans * glm::vec4(position_, 1.0f) );
    orientation_ = glm::mat3(update_trans * glm::mat4(orientation_));

    //TODO: Update this_frame vertex/normal maps for the next iteration

    //Copy the frame for the next update
  } else {
    //TODO: Replace this with a copy constructor
    last_frame_.width = this_frame.width;
    last_frame_.height = this_frame.height;
    cudaMalloc((void**)&last_frame_.vertex, last_frame_.width*last_frame_.height*sizeof(glm::vec3));
    cudaMalloc((void**)&last_frame_.normal, last_frame_.width*last_frame_.height*sizeof(glm::vec3));
    cudaMalloc((void**)&last_frame_.color, last_frame_.width*last_frame_.height*sizeof(Color256));
    has_frame_ = true;
  }

  cudaMemcpy(last_frame_.vertex, this_frame.vertex, this_frame.width*this_frame.height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(last_frame_.normal, this_frame.normal, this_frame.width*this_frame.height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(last_frame_.color, this_frame.color, this_frame.width*this_frame.height*sizeof(Color256), cudaMemcpyDeviceToDevice);
}

//This is based on: http://www.sci.utah.edu/~wallstedt/LU.htm
void RGBDCamera::solveCholesky(const int dimension, const float* A, const float* b, float* x) const {
  //Compute LU decomposition
  std::vector<float> LU(dimension*dimension);
  for (int k = 0; k < dimension; ++k){
    double sum = 0.;
    for (int p = 0; p < k; ++p)
      sum += LU[k*dimension + p] * LU[k*dimension + p];
    LU[k*dimension + k] = sqrt(A[k*dimension + k] - sum);
    for (int i = k + 1; i < dimension; ++i){
      double sum = 0.;
      for (int p = 0; p<k; ++p)sum += LU[i*dimension + p] * LU[k*dimension + p];
      LU[i*dimension + k] = (A[i*dimension + k] - sum) / LU[k*dimension + k];
    }
  }

  //Solve linear equation
  std::vector<float> y(dimension);
  for (int i = 0; i < dimension; ++i){
    double sum = 0.;
    for (int k = 0; k<i; ++k)sum += LU[i*dimension + k] * y[k];
    y[i] = (b[i] - sum) / LU[i*dimension + i];
  }
  for (int i = dimension - 1; i >= 0; --i){
    double sum = 0.;
    for (int k = i + 1; k < dimension; ++k)
      sum += LU[k*dimension + i] * x[k];
    x[i] = (y[i] - sum) / LU[i*dimension + i];
  }
}

} //namespace sensor

} //namespace octree_slam
