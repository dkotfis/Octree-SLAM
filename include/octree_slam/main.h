
#ifndef MAIN_H_
#define MAIN_H_

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>

// GL Dependencies
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/utilities.h>
#include <octree_slam/common_types.h>
#include <octree_slam/rendering/glfw_camera_controller.h>
#include <octree_slam/rendering/opengl_renderer.h>
#include <octree_slam/rendering/cuda_renderer.h>
#include <octree_slam/sensor/openni_device.h>
#include <octree_slam/sensor/image_kernels.h>
#include <octree_slam/sensor/rgbd_camera.h>
#include <octree_slam/world/scene.h>

#define DRAW_CAMERA_COLOR 0
#define DRAW_POINT_CLOUD 0
#define USE_CUDA_RASTERIZER 0
#define VOXELIZE 1
#define OCTREE 0

//-------------------------------
//----------GLOBAL STUFF-----------
//-------------------------------

//Screen/input parameters
int width_ = 1280; int height_ = 960;

//Light position
glm::vec3 lightpos_ = glm::vec3(0, 2.0f, 2.0f);

//Physical camera device interface
octree_slam::sensor::OpenNIDevice* camera_device_;

//Camera localization
octree_slam::sensor::RGBDCamera* camera_estimation_;

//Virtual camera controller
octree_slam::rendering::GLFWCameraController* camera_;

//OpenGL Rendering engine
octree_slam::rendering::OpenGLRenderer* gl_renderer_;

//CUDA Rendering engine
octree_slam::rendering::CUDARenderer* cuda_renderer_;

//Scene object
octree_slam::world::Scene* scene_;
glm::vec3* points_; //TODO: Make this part of scene

int frame_;
int fpstracker_;
double seconds_;
int fps_ = 0;

//Rendering window
GLFWwindow *window_;

int main(int argc, char** argv);

bool init(int argc, char* argv[]);
void mainLoop();
void errorCallback(int error, const char *description);

#endif
