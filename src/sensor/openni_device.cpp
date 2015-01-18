
// Octree-SLAM Dependencies
#include <octree_slam/sensor/openni_device.h>
#include <octree_slam/sensor/openni_kernels.h>

// CUDA Dependencies
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace octree_slam {

namespace sensor {

OpenNIDevice::OpenNIDevice() : depth_width_(0), depth_height_(0), 
  color_width_(0), color_height_(0) {

  //Initialize openni
  checkONIError(openni::OpenNI::initialize());

  //Create the openni device
  device_ = new openni::Device();

  //Open the device
  checkONIError(device_->open(openni::ANY_DEVICE));

  //Create and start video streams
  depth_ = new openni::VideoStream();
  color_ = new openni::VideoStream();
  if (device_->getSensorInfo(openni::SENSOR_DEPTH) != NULL) {
    checkONIError(depth_->create(*device_, openni::SENSOR_DEPTH));
  } else {
    printf("[OpenNIDevice] ERROR: OpenNI Device does not support depth!");
  }
  if (device_->getSensorInfo(openni::SENSOR_COLOR) != NULL) {
    checkONIError(color_->create(*device_, openni::SENSOR_COLOR));
  } else {
    printf("[OpenNIDevice] ERROR: OpenNI Device does not support color!");
  }
  checkONIError(depth_->start());
  checkONIError(color_->start());

  //Initialize sizes with first frame
  openni::VideoFrameRef frame;
  checkONIError(depth_->readFrame(&frame));
  depth_width_ = frame.getWidth();
  depth_height_ = frame.getHeight();
  printf("[OpenNIDevice] Initialized depth to %d by %d. \n", depth_width_, depth_height_);
  checkONIError(color_->readFrame(&frame));
  color_width_ = frame.getWidth();
  color_height_ = frame.getHeight();
  printf("[OpenNIDevice] Initialized color to %d by %d. \n", color_width_, color_height_);

  //Compute focal lengths
  depth_focal_.x = 2.0f * atan(0.5f * (float) depth_width_ / depth_->getHorizontalFieldOfView());
  depth_focal_.y = 2.0f * atan(0.5f * (float) depth_height_ / depth_->getVerticalFieldOfView());
  color_focal_.x = 2.0f * atan(0.5f * (float) color_width_ / color_->getHorizontalFieldOfView());
  color_focal_.y = 2.0f * atan(0.5f * (float) color_height_ / color_->getVerticalFieldOfView());

  //Allocate GPU memory for frames
  cudaMalloc((void**) &d_pixel_depth_, depth_width_*depth_height_*sizeof(openni::DepthPixel));
  cudaMalloc((void**) &d_pixel_color_, color_width_*color_height_*sizeof(openni::RGB888Pixel));
  cudaMalloc((void**) &d_vertex_map_, depth_width_*depth_height_*sizeof(glm::vec3));
  cudaMalloc((void**) &d_normal_map_, depth_width_*depth_height_*sizeof(glm::vec3));
}

OpenNIDevice::~OpenNIDevice() {

  //Turn off streams
  depth_->stop();
  depth_->destroy();
  color_->stop();
  color_->destroy();

  //Close the device
  device_->close();

  //Deallocate memory
  delete depth_;
  delete color_;
  delete device_;

  //Shutdown OpenNI
  openni::OpenNI::shutdown();
  
  //Clean up CUDA memory
  cudaFree(&d_pixel_depth_);
  cudaFree(&d_pixel_color_);
  cudaFree(&d_vertex_map_);
  cudaFree(&d_normal_map_);
}

long long OpenNIDevice::readFrame() {
  //Read a depth frame from the device
  openni::VideoFrameRef frame;
  checkONIError(depth_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM
    && frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM) {
    printf("[OpenNIDevice] ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (depth_width_ != frame.getWidth() || depth_height_ != frame.getHeight()) {
    printf("[OpenNIDevice] ERROR: Received a frame of unexpected size.");
  }

  //Get timestamp from frame
  long long timestamp = (long long)frame.getTimestamp();

  //Get data from frame and copy to GPU
  openni::DepthPixel* pixel_depth = (openni::DepthPixel*) frame.getData();
  cudaMemcpy(d_pixel_depth_, pixel_depth, depth_width_*depth_height_*sizeof(openni::DepthPixel), cudaMemcpyHostToDevice);

  //Generate vertex and normal maps from the data
  generateVertexMap(d_pixel_depth_, d_vertex_map_, depth_width_, depth_height_, depth_focal_);
  generateNormalMap(d_vertex_map_, d_normal_map_, depth_width_, depth_height_);

  //Read a color frame from the device
  checkONIError(color_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888) {
    printf("[OpenNIDevice] ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (color_width_ != frame.getWidth() || color_height_ != frame.getHeight()) {
    printf("[OpenNIDevice] ERROR: Received a frame of unexpected size.");
  }

  //Get data from frame and copy to GPU
  openni::RGB888Pixel* pixel_color = (openni::RGB888Pixel*) frame.getData();
  cudaMemcpy(d_pixel_color_, pixel_color, color_width_*color_height_*sizeof(openni::RGB888Pixel), cudaMemcpyHostToDevice);

  return timestamp;
}

void OpenNIDevice::initPBO() {

  glGenTextures(1, &displayImage_);
  glBindTexture(GL_TEXTURE_2D, displayImage_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, color_width_, color_height_, 0, GL_BGRA,
    GL_UNSIGNED_BYTE, NULL);

  // set up vertex data parameter
  int num_texels = color_width_*color_height_;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo_);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo_);

}

void OpenNIDevice::drawColor() const {

  uchar4 *dptr;
  cudaGLMapBufferObject((void**)&dptr, pbo_);

  //Convert color frame to pbo format
  writeColorToPBO(d_pixel_color_, dptr, color_width_*color_height_);

  cudaGLUnmapBufferObject(pbo_);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, displayImage_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, color_width_, color_height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

}

void OpenNIDevice::checkONIError(const openni::Status& error) {
  if (error != openni::STATUS_OK) {
    printf("[OpenNIDevice] Received ONI Error Message: %s \n", openni::OpenNI::getExtendedError());
  }
}

} //namespace sensor

} //namespace octree_slam
