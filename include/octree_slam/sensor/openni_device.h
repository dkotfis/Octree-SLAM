#ifndef OPENNI_DEVICE_H_
#define OPENNI_DEVICE_H_

// OpenGL Dependency
#include <GL/glew.h>
#include <glm/glm.hpp>

// OpenNI Dependency
#include <OpenNI.h>

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

  //Function for initializing a pixel buffer object for rendering debug
  void initPBO();

  //Function for drawing the current color frame to the screen
  //Requires initPBO to have been called first
  void drawColor() const;

  //Accessor method to the GPU depth data
  const openni::DepthPixel* depthFrameGPU() const { return d_pixel_depth_; };

  //Accessor method to the GPU color data
  const openni::RGB888Pixel* colorFrameGPU() const { return d_pixel_color_; };

  //Accessor method to the depth frame width
  const int depthFrameWidth() const { return depth_width_; };

  //Accessor method to the depth frame height
  const int depthFrameHeight() const { return depth_height_; };

  //Accessor method to the color frame width
  const int colorFrameWidth() const { return color_width_; };

  //Accessor method to the color frame height
  const int colorFrameHeight() const { return color_height_; };

  //Accessor to the PBO to render
  const GLuint pbo() const { return pbo_; };

private:

  //Error checking function to print out ONI calls that are not successful
  void checkONIError(const openni::Status& error);

  //OpenNI device object
  openni::Device* device_;

  //OpenNI streams
  openni::VideoStream* depth_;
  openni::VideoStream* color_;

  //Sizes of frames
  int depth_width_, depth_height_, color_width_, color_height_;

  //Focal lenfths
  glm::vec2 depth_focal_;
  glm::vec2 color_focal_;

  //Current frame data in GPU memory
  openni::DepthPixel* d_pixel_depth_;
  openni::RGB888Pixel* d_pixel_color_;
  glm::vec3* d_vertex_map_;
  glm::vec3* d_normal_map_;

  //PBO for drawing images to the screen
  GLuint pbo_ = (GLuint) NULL;

}; //class OpenNIDevice

} //namespace sensor

} //namespace octree_slam


#endif //OPENNI_DEVICE_H_

