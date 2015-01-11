
#ifndef OPENNI_DEVICE_H_
#define OPENNI_DEVICE_H_

#include <OpenNI.h>

namespace octree_slam {

namespace sensor {

class OpenNIDevice {

public:

  //Default Constructor
  //Activates an RGBD sensor through OpenNI
  OpenNIDevice();

  //Destructor
  ~OpenNIDevice();

  //Function for receiving the current frame from the device
  //Return timestamp
  long long readFrame();

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
    
}; //class OpenNIDevice

} //namespace sensor

} //namespace octree_slam


#endif //OPENNI_DEVICE_H_

