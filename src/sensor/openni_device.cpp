
// Octree-SLAM Dependencies
#include <octree_slam/sensor/openni_device.h>
#include <octree_slam/sensor/image_kernels.h>

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

  //Turn off autoexposure and white balancing
  checkONIError(color_->getCameraSettings()->setAutoExposureEnabled(false));
  checkONIError(color_->getCameraSettings()->setAutoWhiteBalanceEnabled(false));

  //Turn on color/depth synchronization
  device_->setDepthColorSyncEnabled(true);
  device_->setImageRegistrationMode(openni::ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

  //Start video streams
  checkONIError(depth_->start());
  checkONIError(color_->start());

  //Initialize sizes with first frame
  openni::VideoFrameRef frame;
  checkONIError(depth_->readFrame(&frame));
  raw_frame_ = new RawFrame(frame.getWidth(), frame.getHeight());
  printf("[OpenNIDevice] Initialized frame to %d by %d. \n", raw_frame_->width, raw_frame_->height);
  //TODO: Check whether the color frame is the same size, and do something if it is not
  /*
  checkONIError(color_->readFrame(&frame));
  color_width_ = frame.getWidth();
  color_height_ = frame.getHeight();
  printf("[OpenNIDevice] Initialized color to %d by %d. \n", color_width_, color_height_);
  */

  //Compute focal lengths
  depth_focal_.x = (float) raw_frame_->width / (2.0f * tan(0.5f * depth_->getHorizontalFieldOfView()));
  depth_focal_.y = (float) raw_frame_->height / (2.0f * tan(0.5f * depth_->getVerticalFieldOfView()));

  //Get the framerate
  fps_ = frame.getVideoMode().getFps();

  //Initialize the clock
  time_ = clock();
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
  
  delete raw_frame_;
}

long long OpenNIDevice::readFrame() {

  //Don't do anything if a new frame isn't going to be ready
  if ((float) (clock() - time_)/1000.0f < 1.0f/(float)fps_) {
    return timestamp_;
  } else {
    time_ = clock();
  }

  //Read a depth frame from the device
  openni::VideoFrameRef frame;
  checkONIError(depth_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM
    && frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM) {
    printf("[OpenNIDevice] ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (raw_frame_->width != frame.getWidth() || raw_frame_->height != frame.getHeight()) {
    printf("[OpenNIDevice] ERROR: Received a frame of unexpected size.");
  }

  //Get data from frame and copy to GPU
  openni::DepthPixel* pixel_depth = (openni::DepthPixel*) frame.getData();
  cudaMemcpy(raw_frame_->depth, pixel_depth, raw_frame_->width*raw_frame_->height*sizeof(openni::DepthPixel), cudaMemcpyHostToDevice);

  //Get timestamp from frame
  timestamp_ = (long long)frame.getTimestamp();

  //Read a color frame from the device
  checkONIError(color_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888) {
    printf("[OpenNIDevice] ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (raw_frame_->width != frame.getWidth() || raw_frame_->height != frame.getHeight()) {
    printf("[OpenNIDevice] ERROR: Received a frame of unexpected size.");
  }

  //TODO: Check the timestamp of the color frame

  //Get data from frame and copy to GPU
  openni::RGB888Pixel* pixel_color = (openni::RGB888Pixel*) frame.getData();
  cudaMemcpy(raw_frame_->color, pixel_color, raw_frame_->width*raw_frame_->height*sizeof(openni::RGB888Pixel), cudaMemcpyHostToDevice);

  //Timestamp the data
  raw_frame_->timestamp = timestamp_;

  return timestamp_;
}

void OpenNIDevice::checkONIError(const openni::Status& error) {
  if (error != openni::STATUS_OK) {
    printf("[OpenNIDevice] Received ONI Error Message: %s \n", openni::OpenNI::getExtendedError());
  }
}

} //namespace sensor

} //namespace octree_slam
