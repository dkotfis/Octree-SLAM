#ifndef SCENE_STRUCTS_H_
#define SCENE_STRUCTS_H_

#include <stdint.h>
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

struct Camera {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 projection;
  glm::mat4 modelview;
  glm::mat4 mvp;
};

struct Color256 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct Frame {
  Color256* color;
  glm::vec3* vertex;
  glm::vec3* normal;
  int width;
  int height;
};

#endif //SCENE_STRUCTS_H_
