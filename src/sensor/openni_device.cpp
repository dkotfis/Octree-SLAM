
#include <octree_slam/sensor/openni_device.h>

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
    printf("ERROR: OpenNI Device does not support depth!");
  }
  //TODO: Structure doesn't support RGB, sigh... figure out what to do now :-(
  if (device_->getSensorInfo(openni::SENSOR_IR) != NULL) {
    checkONIError(color_->create(*device_, openni::SENSOR_IR));
  } else {
    printf("ERROR: OpenNI Device does not support color!");
  }
  checkONIError(depth_->start());
  checkONIError(color_->start());

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
  
}

long long OpenNIDevice::readFrame() {
  //Read a depth frame from the device
  openni::VideoFrameRef frame;
  checkONIError(depth_->readFrame(&frame));

  //Verify format of frame
  if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM
    && frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM) {
    printf("ERROR: Unexpected frame format! \n");
  }

  //Verify size of frame
  if (depth_width_ == 0) {
    depth_width_ = frame.getWidth();
    depth_height_ = frame.getHeight();
  }
  if (depth_width_ != frame.getWidth() || depth_height_ != frame.getHeight()) {
    printf("ERROR: Received a frame of unexpected size.");
  }

  //Get timestamp from frame
  long long timestamp = (long long)frame.getTimestamp();

  //Get data from frame
  openni::DepthPixel* pixel_depth = (openni::DepthPixel*) frame.getData();

  return timestamp;
}

void OpenNIDevice::checkONIError(const openni::Status& error) {
  if (error != openni::STATUS_OK) {
    printf("%s \n", openni::OpenNI::getExtendedError());
  }
}

} //namespace sensor

} //namespace octree_slam
