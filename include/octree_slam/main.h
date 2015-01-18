
#ifndef MAIN_H
#define MAIN_H

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
#include <glm/gtc/type_ptr.hpp>
#include <glslUtil/glslUtility.hpp>
#include <glm/gtc/matrix_transform.hpp>

// CUDA Dependencies
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Octree-SLAM Dependencies
#include <octree_slam/utilities.h>
#include <octree_slam/scene_structs.h>
#include <octree_slam/voxelization/voxelization.h>
#include <octree_slam/svo/svo.h>
#include <octree_slam/rendering/rasterize_kernels.h>
#include <octree_slam/sensor/openni_device.h>

#define DRAW_CAMERA_COLOR 1
#define DRAW_CAMERA_DEPTH 0
#define USE_CUDA_RASTERIZER 0
#define VOXELIZE 0
#define OCTREE 0

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

GLFWwindow *window;

obj* mesh;
vector <obj*> meshes;

bmp_texture tex;

//Voxelized mesh
Mesh m_vox;

int mode=0;
bool barycenter = false;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
float* nbo;
int nbosize;
vector<glm::vec4>* texcoord;

// Uniform locations for the GL shaders
GLuint mvp_location;
GLuint proj_location;
GLuint norm_location;
GLuint light_location;

// VAO's for the GL pipeline
GLuint buffers[3];

//-------------------------------
//----------GLOBAL STUFF-----------
//-------------------------------

//Screen/input parameters
int width = 800; int height = 800;
glm::vec3 position(0.0f, 0.0f, 3.0f);
float horizontalAngle = 3.14f;
float verticalAngle = 0.0f;
float FoV = 60.0f;
float zNear = 0.001;
float zFar = 10000.0;
float speed = 1.0f;
float mouseSpeed = 0.005f;
double lastTime;

//Camera matrices
glm::mat4 projection;
glm::mat4 model;
glm::mat4 view;
glm::mat4 modelview;
glm::mat4 mvp;

//Light position
glm::vec3 lightpos = glm::vec3(0, 2.0f, 2.0f);

//Physical camera device interface
octree_slam::sensor::OpenNIDevice camera_device_;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);
void loadMultipleObj(int choice, int type);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();
void runGL();

void computeMatricesFromInputs();

#ifdef __APPLE__
	void display();
#else
	void display();
	void keyboard(unsigned char key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------
bool init(int argc, char* argv[]);

void voxelizeScene();

//CUDA Rasterizer Setup
void initCudaPBO();
void initCuda();
void initCudaTextures();
void initCudaVAO();
GLuint initPassthroughShaders();

//GL Rasterizer Setup
void initGL();
GLuint initDefaultShaders();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void mainLoop();
void errorCallback(int error, const char *description);
bool LB = false;
void MouseClickCallback(GLFWwindow *window, int button, int action, int mods);
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

#endif
