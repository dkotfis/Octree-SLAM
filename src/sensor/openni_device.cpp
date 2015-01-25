
// Octree-SLAM Dependencies
#include <octree_slam/sensor/openni_device.h>
#include <octree_slam/sensor/openni_kernels.h>

// CUDA Dependencies
#include <cuda_runtime.h>

namespace octree_slam {

namespace sensor {

OpenNIDevice::OpenNIDevice() {

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
  frame_.width = frame.getWidth();
  frame_.height = frame.getHeight();
  printf("[OpenNIDevice] Initialized frame to %d by %d. \n", frame_.width, frame_.height);
  //TODO: Check whether the color frame is the same size, and do something if it is not
  /*
  checkONIError(color_->readFrame(&frame));
  color_width_ = frame.getWidth();
  color_height_ = frame.getHeight();
  printf("[OpenNIDevice] Initialized color to %d by %d. \n", color_width_, color_height_);
  */

  //Compute focal lengths
  depth_focal_.x = (float) frame_.width / (2.0f * tan(0.5f * depth_->getHorizontalFieldOfView()));
  depth_focal_.y = (float) frame_.height / (2.0f * tan(0.5f * depth_->getVerticalFieldOfView()));

  //Allocate GPU memory for frames
  cudaMalloc((void**) &frame_.color, frame_.width*frame_.height*sizeof(Color256));
  cudaMalloc((void**) &frame_.vertex, frame_.width*frame_.height*sizeof(glm::vec3));
  cudaMalloc((void**) &frame_.normal, frame_.width*frame_.height*sizeof(glm::vec3));
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
  cudaFree(frame_.color);
  cudaFree(frame_.vertex);
  cudaFree(frame_.normal);
}

long long OpenNIDevice::readFrame() {
  //Temporarily allocate GPU memory for the raw depth frame
  cudaMalloc((void**)&d_pixel_depth_, frame_.width*frame_.height*sizeof(openni::DepthPixel));

  //Read a depth frame from the device
  openni::VideoFrameRef frame;
  checkONIError(depth_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM
    && frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM) {
    printf("[OpenNIDevice] ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (frame_.width != frame.getWidth() || frame_.height != frame.getHeight()) {
    printf("[OpenNIDevice] ERROR: Received a frame of unexpected size.");
  }

  //Get timestamp from frame
  //TODO: Skip the frame if we've already seen this timestamp
  long long timestamp = (long long)frame.getTimestamp();

  //Get data from frame and copy to GPU
  openni::DepthPixel* pixel_depth = (openni::DepthPixel*) frame.getData();
  cudaMemcpy(d_pixel_depth_, pixel_depth, frame_.width*frame_.height*sizeof(openni::DepthPixel), cudaMemcpyHostToDevice);

  //Generate vertex and normal maps from the data
  generateVertexMap(d_pixel_depth_, frame_.vertex, frame_.width, frame_.height, depth_focal_);
  generateNormalMap(frame_.vertex, frame_.normal, frame_.width, frame_.height);

  //Read a color frame from the device
  checkONIError(color_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888) {
    printf("[OpenNIDevice] ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (frame_.width != frame.getWidth() || frame_.height != frame.getHeight()) {
    printf("[OpenNIDevice] ERROR: Received a frame of unexpected size.");
  }

  //TODO: Check the timestamp of the color frame

  //Get data from frame and copy to GPU
  openni::RGB888Pixel* pixel_color = (openni::RGB888Pixel*) frame.getData();
  cudaMemcpy(frame_.color, pixel_color, frame_.width*frame_.height*sizeof(openni::RGB888Pixel), cudaMemcpyHostToDevice);

  //Free up GPU memory now that the depth frame is not needed 
  cudaFree(d_pixel_depth_);

  return timestamp;
}

void OpenNIDevice::checkONIError(const openni::Status& error) {
  if (error != openni::STATUS_OK) {
    printf("[OpenNIDevice] Received ONI Error Message: %s \n", openni::OpenNI::getExtendedError());
  }
}

} //namespace sensor

} //namespace octree_slam
