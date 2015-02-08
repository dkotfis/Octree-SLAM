
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

const int RGBDCamera::PYRAMID_ITERS[] = {10, 5, 4};
const float RGBDCamera::W_RGBD = 0.1;
const float RGBDCamera::MOVE_THRESH = 0.01;
const float RGBDCamera::TURN_THRESH = 0.05;

RGBDCamera::RGBDCamera(const int width, const int height, const glm::vec2 &focal_length) : 
  width_(width), height_(height), focal_length_(focal_length), pass_(0) {
}

RGBDCamera::~RGBDCamera() {
  if (pass_ >= 1) {
    for (int i = 0; i < PYRAMID_DEPTH; i++) {
      delete last_icp_frame_[i];
      delete last_rgbd_frame_[i];
    }
  }
  if (pass_ >= 2) {
    for (int i = 0; i < PYRAMID_DEPTH; i++) {
      delete current_icp_frame_[i];
      delete current_rgbd_frame_[i];
    }
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
  cudaMalloc((void**)&filtered_depth, this_frame->width*this_frame->height*sizeof(uint16_t));
  bilateralFilter(this_frame->depth, filtered_depth, this_frame->width, this_frame->height);

  //Convert the input color data to intensity
  float* temp_intensity;
  cudaMalloc((void**)&temp_intensity, this_frame->width*this_frame->height*sizeof(float));
  colorToIntensity(this_frame->color, temp_intensity, this_frame->width*this_frame->height);

  //Create pyramids
  for (int i = 0; i < PYRAMID_DEPTH; i++) {
    //Fill in sizes the first two times through
    if (pass_ < 2) {
      current_icp_frame_[i] = new ICPFrame(this_frame->width/pow(2,i), this_frame->height/pow(2,i));
      current_rgbd_frame_[i] = new RGBDFrame(this_frame->width/pow(2,i), this_frame->height/pow(2,i));
    }

    //Add ICP data
    generateVertexMap(filtered_depth, current_icp_frame_[i]->vertex, current_icp_frame_[i]->width, current_icp_frame_[i]->height, focal_length_, make_int2(this_frame->width, this_frame->height));
    generateNormalMap(current_icp_frame_[i]->vertex, current_icp_frame_[i]->normal, current_icp_frame_[i]->width, current_icp_frame_[i]->height);

    //Add RGBD data
    cudaMemcpy(current_rgbd_frame_[i]->vertex, current_icp_frame_[i]->vertex, current_rgbd_frame_[i]->width*current_rgbd_frame_[i]->height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(current_rgbd_frame_[i]->intensity, temp_intensity, current_rgbd_frame_[i]->width*current_rgbd_frame_[i]->height*sizeof(float), cudaMemcpyDeviceToDevice);

    //Downsample depth and color if not the last iteration
    if (i != (PYRAMID_DEPTH-1)) {
      subsampleDepth(filtered_depth, current_icp_frame_[i]->width, current_icp_frame_[i]->height);
      subsample(temp_intensity, current_rgbd_frame_[i]->width, current_rgbd_frame_[i]->height);
      cudaDeviceSynchronize();
    }
  }

  //Clear the filtered depth and temporary color since they are no longer needed
  cudaFree(filtered_depth);
  cudaFree(temp_intensity);

  if (pass_ >= 1) {
    glm::mat4 update_trans(1.0f);

    //Loop through pyramids backwards (coarse first)
    for (int i = PYRAMID_DEPTH - 1; i >= 0; i--) {

      //Get a copy of the ICP frame for this pyramid level
      ICPFrame icp_f(current_icp_frame_[i]->width, current_icp_frame_[i]->height);
      cudaMemcpy(icp_f.vertex, current_icp_frame_[i]->vertex, icp_f.width*icp_f.height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
      cudaMemcpy(icp_f.normal, current_icp_frame_[i]->normal, icp_f.width*icp_f.height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

      //Get a copy of the RGBD frame for this pyramid level
      //RGBDFrame rgbd_f(current_rgbd_frame_[i]->width, current_rgbd_frame_[i]->height);
      //cudaMemcpy(rgbd_f.vertex, current_rgbd_frame_[i]->vertex, rgbd_f.width*rgbd_f.height*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
      //cudaMemcpy(rgbd_f.intensity, current_rgbd_frame_[i]->intensity, rgbd_f.width*rgbd_f.height*sizeof(float), cudaMemcpyDeviceToDevice);

      //Apply the most recent update to the points/normals
      if (i < (PYRAMID_DEPTH-1)) {
        transformVertexMap(icp_f.vertex, update_trans, icp_f.width*icp_f.height);
        transformNormalMap(icp_f.normal, update_trans, icp_f.width*icp_f.height);
        cudaDeviceSynchronize();
      }

      //Loop through iterations
      for (int j = 0; j < PYRAMID_ITERS[i]; j++) {

        //Get the Geometric ICP cost values
        float A1[6 * 6];
        float b1[6];
        computeICPCost2(last_icp_frame_[i], icp_f, A1, b1);

        //Get the Photometric RGB-D cost values
        //float A2[6*6];
        //float b2[6];
        //compueRGBDCost(last_rgbd_frame_, rgbd_f, A2, b2);

        //Combine the two
        //for (size_t k = 0; k < 6; k++) {
          //for (size_t l = 0; l < 6; l++) {
            //A1[6 * k + l] += A2[6 * k + l];
          //}
          //b1[k] += b2[k];
        //}

        //Solve for the optimized camera transformation
        float x[6];
        solveCholesky(6, A1, b1, x);

        //Check for NaN/divergence
        if (isnan(x[0]) || isnan(x[1]) || isnan(x[2]) || isnan(x[3]) || isnan(x[4]) || isnan(x[5])) {
          printf("Camera tracking is lost.\n");
          break;
        }

        //Update position/orientation of the camera
        glm::mat4 this_trans = 
            glm::rotate(glm::mat4(1.0f), -x[2] * 180.0f / 3.14159f, glm::vec3(0.0f, 0.0f, 1.0f)) 
          * glm::rotate(glm::mat4(1.0f), -x[1] * 180.0f / 3.14159f, glm::vec3(0.0f, 1.0f, 0.0f))
          * glm::rotate(glm::mat4(1.0f), -x[0] * 180.0f / 3.14159f, glm::vec3(1.0f, 0.0f, 0.0f)) 
          * glm::translate(glm::mat4(1.0f), glm::vec3(x[3], x[4], x[5]));

        update_trans = this_trans * update_trans;
        
        //Apply the update to the points/normals
        if (j < (PYRAMID_ITERS[i] - 1)) {
          transformVertexMap(icp_f.vertex, this_trans, icp_f.width*icp_f.height);
          transformNormalMap(icp_f.normal, this_trans, icp_f.width*icp_f.height);
          cudaDeviceSynchronize();
        }

      }
    }
    //Update the global transform with the result
    position_ = glm::vec3(glm::vec4(position_, 1.0f) * update_trans);
    orientation_ = glm::mat3(glm::mat4(orientation_) * update_trans);
  }

  if (pass_ < 2) {
    pass_++;
  }

  //Swap current and last frames
  for (int i = 0; i < PYRAMID_DEPTH; i++) {
    ICPFrame* temp = current_icp_frame_[i];
    current_icp_frame_[i] = last_icp_frame_[i];
    last_icp_frame_[i] = temp;
    //TODO: Longterm, only RGBD should do this. ICP should not swap, as last_frame should be updated by a different function
    RGBDFrame* temp2 = current_rgbd_frame_[i];
    current_rgbd_frame_[i] = last_rgbd_frame_[i];
    last_rgbd_frame_[i] = temp2;
  }

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
