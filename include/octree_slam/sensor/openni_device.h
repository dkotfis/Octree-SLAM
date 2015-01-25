#ifndef OPENNI_DEVICE_H_
#define OPENNI_DEVICE_H_

// OpenGL Dependency
#include <GL/glew.h>
#include <glm/glm.hpp>

// OpenNI Dependency
#include <OpenNI.h>

// Octree-SLAM Dependency
#include <octree_slam/common_types.h>

/* Uncomment these if the device does not support color
#define RGB888Pixel Grayscale16Pixel
#define SENSOR_COLOR SENSOR_IR
#define PIXEL_FORMAT_RGB888 PIXEL_FORMAT_GRAY16
*/

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

  //Accessor method to the current data frame
  const Frame frame() const { return frame_; };

  //Accessor method to the frame width
  const int frameWidth() const { return frame_.width; };

  //Accessor method to the frame height
  const int frameHeight() const { return frame_.height; };

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

  //Camera frame data
  Frame frame_;

  //Focal length
  glm::vec2 depth_focal_;

  //Current frame data in GPU memory
  openni::DepthPixel* d_pixel_depth_;

}; //class OpenNIDevice

} //namespace sensor

} //namespace octree_slam


#endif //OPENNI_DEVICE_H_

