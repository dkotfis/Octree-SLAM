#ifndef SCENE_STRUCTS_H_
#define SCENE_STRUCTS_H_

#include <stdint.h>
#include "glm/glm.hpp"
#include <vector_types.h>

struct BoundingBox {
  glm::vec3 bbox0 = glm::vec3(0.0f);
  glm::vec3 bbox1 = glm::vec3(0.0f);

  //Checks whether a bounding box contains the other
  bool contains(const BoundingBox& other) const;

  //Gets the max distance that a bounding box sits outside the other (in a single dimension)
  float distanceOutside(const BoundingBox& other) const;
};

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
  BoundingBox bbox;
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
  float fov;
};

struct Color256 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct VoxelGrid {
  ~VoxelGrid();
  void setBoundingBox(const float3& b0, const float3& b1);
  glm::vec4* centers;
  glm::vec4* colors;
  int size = 0;
  float scale = 0.0f;
  BoundingBox bbox;
};

struct RawFrame {
  RawFrame(const int w, const int h);
  ~RawFrame();
  Color256* color;
  uint16_t* depth;
  int height;
  int width;
  long long timestamp;
};

struct SVO {
  unsigned int* data;
  glm::vec3 center;
  float size;
};

#endif //SCENE_STRUCTS_H_
