
#ifndef MAIN_H_
#define MAIN_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <objUtil/objloader.h>

// GL Dependencies
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/utilities.h>
#include <octree_slam/scene_structs.h>
#include <octree_slam/voxelization/voxelization.h>
#include <octree_slam/svo/svo.h>
#include <octree_slam/rendering/glfw_camera_controller.h>
#include <octree_slam/rendering/opengl_renderer.h>
#include <octree_slam/rendering/cuda_renderer.h>
#include <octree_slam/sensor/openni_device.h>

#define DRAW_CAMERA_COLOR 0
#define DRAW_CAMERA_DEPTH 0
#define USE_CUDA_RASTERIZER 0
#define VOXELIZE 0
#define OCTREE 0

//-------------------------------
//----------GLOBAL STUFF-----------
//-------------------------------

//Screen/input parameters
int width_ = 800; int height_ = 800;

//Light position
glm::vec3 lightpos_ = glm::vec3(0, 2.0f, 2.0f);

//Physical camera device interface
octree_slam::sensor::OpenNIDevice* camera_device_;

//Virtual camera controller
octree_slam::rendering::GLFWCameraController* camera_;

//OpenGL Rendering engine
octree_slam::rendering::OpenGLRenderer* gl_renderer_;

//CUDA Rendering engine
octree_slam::rendering::CUDARenderer* cuda_renderer_;

int frame_;
int fpstracker_;
double seconds_;
int fps_ = 0;

//Rendering window
GLFWwindow *window_;

//Texture information
bmp_texture tex_;

//Mesh geometry representation
obj* mesh_;
vector <obj*> meshes_;
Mesh mesh_geom_;

int main(int argc, char** argv);

bool init(int argc, char* argv[]);
void mainLoop();
void errorCallback(int error, const char *description);

void loadMultipleObj(int choice, int type);
void readBMP(const char* filename, bmp_texture &tex);
Mesh buildScene();

#endif
