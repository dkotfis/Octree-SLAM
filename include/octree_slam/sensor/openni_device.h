#ifndef OPENNI_DEVICE_H_
#define OPENNI_DEVICE_H_

#include <time.h>

// OpenGL Dependency
#include <GL/glew.h>
#include <glm/glm.hpp>

// OpenNI Dependency
#include <OpenNI.h>

// Octree-SLAM Dependency
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace sensor {

class OpenNIDevice {

public:

  //Default Constructor
  //Activates an RGBD sensor through OpenNI
  OpenNIDevice();

  //Destructor
  ~OpenNIDevice();

  //Function for receiving the current frame from the device.
  //It is copyied to GPU, and vertex/normal maps are generated.
  //Return timestamp
  long long readFrame();

  //Accessor method to get the current raw frame data
  const RawFrame* rawFrame() const { return raw_frame_; };

  //Accessor method to the frame width
  const int frameWidth() const { return raw_frame_->width; };

  //Accessor method to the frame height
  const int frameHeight() const { return raw_frame_->height; };

  //Accessor method to the focal length
  const glm::vec2& focalLength() const { return depth_focal_; };

private:

  //Error checking function to print out ONI calls that are not successful
  void checkONIError(const openni::Status& error);

  //OpenNI device object
  openni::Device* device_;

  //OpenNI streams
  openni::VideoStream* depth_;
  openni::VideoStream* color_;

  //Focal length
  glm::vec2 depth_focal_;

  //The framerate of the camera stream
  int fps_;

  //Current frame data in GPU memory
  RawFrame* raw_frame_;

  //The most recent frame timestamp
  long long timestamp_;

  //The most recent frame time
  clock_t time_;

}; //class OpenNIDevice

} //namespace sensor

} //namespace octree_slam


#endif //OPENNI_DEVICE_H_

