
#include <vector>

// Octree-SLAM Dependencies
#include <octree_slam/sensor/rgbd_camera.h>
#include <octree_slam/sensor/localization_kernels.h>
#include <octree_slam/sensor/image_kernels.h>

// OpenGL Dependency
#include <glm/gtc/matrix_transform.hpp>

// CUDA Dependencies
#include <cuda_runtime.h>

namespace octree_slam {

namespace sensor {

const int RGBDCamera::PYRAMID_ITERS[] = {4, 5, 10};
const float RGBDCamera::W_RGBD = 0.1;

RGBDCamera::RGBDCamera(const int width, const int height, const glm::vec2 &focal_length) : 
  width_(width), height_(height), focal_length_(focal_length), has_frame_(false) {
}

RGBDCamera::~RGBDCamera() {
  if (has_frame_) {
    delete last_icp_frame_;
    delete last_rgbd_frame_;
  }
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

void RGBDCamera::update(const RawFrame* this_frame) {
  //Apply bilateral filter to incoming depth
  uint16_t* filtered_depth;
  cudaMalloc((void**)&filtered_depth, this_frame->width*this_frame->height);
  bilateralFilter(this_frame->depth, filtered_depth, this_frame->width, this_frame->height);

  //Generate the vertex and normal maps for ICP
  ICPFrame icp_f(this_frame->width, this_frame->height);
  generateVertexMap(filtered_depth, icp_f.vertex, this_frame->width, this_frame->height, focal_length_);
  generateNormalMap(icp_f.vertex, icp_f.normal, this_frame->width, this_frame->height);
  cudaMemcpy(icp_f.color, this_frame->color, this_frame->width*this_frame->height*sizeof(Color256), cudaMemcpyDeviceToDevice);
  //TODO: Create pyramids

  //Clear the filtered depth since it is no longer needed
  cudaFree(filtered_depth);

  if (has_frame_) {
    //TODO: Loop over pyramids

    //Get the Geometric ICP cost values
    float A1[6*6];
    float b1[6];
    computeICPCost(last_icp_frame_, icp_f, A1, b1);

    //Get the Photometric RGB-D cost values
    //float A2[6*6];
    //float b2[6];
    //compueRGBDCost(last_rgbd_frame_, rgbd_f, A2, b2);

    //Solve for the optimized camera transformation
    float x[6];
    solveCholesky(6, A1, b1, x);

    //Update position/orientation of the camera
    glm::mat4 update_trans = glm::mat4(glm::mat3(1.0f, x[2], -x[1], -x[2], 1.0f, x[0], x[1], -x[0], 1.0f)) 
      * glm::translate(glm::mat4(1.0f), glm::vec3(x[3], x[4], x[5]));
    position_ = glm::vec3( update_trans * glm::vec4(position_, 1.0f) );
    orientation_ = glm::mat3(update_trans * glm::mat4(orientation_));

    //TODO: Update this_frame vertex/normal maps for the next iteration

    //Copy the frame for the next update
  } else {
    last_icp_frame_ = new ICPFrame(this_frame->width, this_frame->height);
    last_rgbd_frame_ = new RGBDFrame(this_frame->width, this_frame->height);
    has_frame_ = true;
  }

  //Store ICP data for the next frame
  cudaMemcpy(last_icp_frame_->vertex, icp_f.vertex, this_frame->width*this_frame->height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(last_icp_frame_->normal, icp_f.normal, this_frame->width*this_frame->height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(last_icp_frame_->color, icp_f.color, this_frame->width*this_frame->height*sizeof(float), cudaMemcpyDeviceToDevice);
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
