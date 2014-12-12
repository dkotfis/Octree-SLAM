
#ifndef MAIN_H
#define MAIN_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glslUtil/glslUtility.hpp>
#include <iostream>
#include <objUtil/objloader.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <glm/gtc/matrix_transform.hpp>

#define MOUSE_SPEED 2.0*0.0001f
#define ZOOM_SPEED 8
#define MIDDLE_SPEED 12

#define USE_CUDA_RASTERIZER 0
#define VOXELIZE 1
#define OCTREE 1

#include "rasterizeKernels.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "voxelization.h"
#include "svo.h"

using namespace std;

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

int width = 800; int height = 800;
glm::vec3 eye(0.0f, 0.8f, 3.0f);
glm::vec3 center(0.0f, 0.4f, 0.0f);
float zNear = 0.001;
float zFar = 10000.0;
glm::mat4 projection = glm::perspective(60.0f, (float)(width) / (float)(height), zNear, zFar);
glm::mat4 model = glm::mat4();
glm::mat4 view = glm::lookAt(eye,center , glm::vec3(0, 1, 0));
glm::mat4 modelview = view * glm::mat4();
glm::vec3 lightpos = glm::vec3(0, 2.0f, 2.0f);

//Make the camera move on the surface of sphere
float vPhi = 0.0f;
float vTheta = 3.14105926f / 2.0f;
float R = glm::length(eye);

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
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
//Add mouse control
double MouseX = 0.0;
double MouseY = 0.0;
bool LB = false;
bool RB = false;
bool MB = false;
bool inwindow = false;
void MouseClickCallback(GLFWwindow *window, int button, int action, int mods);
void CursorCallback(GLFWwindow *window, double x,double y);
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void CursorEnterCallback(GLFWwindow *window,int entered);

#endif
