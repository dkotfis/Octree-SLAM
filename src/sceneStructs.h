#ifndef SCENE_STRUCTS_H
#define SCENE_STRUCTS_H

#include "glm/glm.hpp"

//This is a lighter weight version of obj
struct Mesh {
  int vbosize;
  int nbosize;
  int cbosize;
  int ibosize;
  int tbosize;
  float* vbo;
  float* nbo;
  float* cbo;
  int* ibo;
  float* tbo;
};

struct bmp_texture {
  glm::vec3 *data;
  int width;
  int height;
};

//Handy function for clamping between two values;
__host__ __device__ inline float clamp(float val, float min, float max) {
  if (val < min) {
    val = min;
  }
  else if (val > max) {
    val = max;
  }
  return val;
}

#endif