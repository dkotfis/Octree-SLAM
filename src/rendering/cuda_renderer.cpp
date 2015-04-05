
#include <string>

// OpenGL Dependencies
#include <GL/glew.h>
#include <glslUtil/glslUtility.hpp>
#include <glm/gtc/matrix_transform.hpp>

// CUDA Dependencies
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Octree-SLAM Dependency
#include <octree_slam/rendering/cuda_renderer.h>
#include <octree_slam/rendering/rasterize_kernels.h>
#include <octree_slam/rendering/cone_tracing_kernels.h>

namespace octree_slam {

namespace rendering {

float CUDARenderer::newcbo_[9] = { 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 };

CUDARenderer::CUDARenderer(const bool voxelize, const std::string& path_prefix, const int width, const int height) : voxelized_(voxelize), width_(width), height_(height), pbo_((GLuint)NULL), frame_(0) {
  //Init textures
  glGenTextures(1, &displayImage_);
  glBindTexture(GL_TEXTURE_2D, displayImage_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

  //Init VAO
  GLfloat vertices[] =
  {
    -1.0f, -1.0f,
    1.0f, -1.0f,
    1.0f, 1.0f,
    -1.0f, 1.0f,
  };

  GLfloat texcoords[] =
  {
    1.0f, 1.0f,
    0.0f, 1.0f,
    0.0f, 0.0f,
    1.0f, 0.0f
  };

  GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

  GLuint vertexBufferObjID[3];
  glGenBuffers(3, vertexBufferObjID);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)positionLocation_, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(positionLocation_);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)texcoordsLocation_, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(texcoordsLocation_);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  // Use device with highest Gflops/s
  cudaGLSetGLDevice(0);

  // set up vertex data parameter
  int num_texels = width*height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo_);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo_);

  // Create passthrough shaders
  const char *attribLocations[] = { "Position", "Tex" };
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;

  glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1)
  {
    glUniform1i(location, 0);
  }

  glActiveTexture(GL_TEXTURE0);
}

CUDARenderer::~CUDARenderer() {
  if (pbo_) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(pbo_);

    glBindBuffer(GL_ARRAY_BUFFER, pbo_);
    glDeleteBuffers(1, &pbo_);

    pbo_ = (GLuint)NULL;
  }
  if (displayImage_) {
    glDeleteTextures(1, &displayImage_);
    displayImage_ = (GLuint)NULL;
  }
  octree_slam::rendering::kernelCleanup();
}

void CUDARenderer::rasterize(const Mesh& geometry, const bmp_texture& texture, const Camera& camera, const glm::vec3& light) {

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  uchar4 *dptr = NULL;
  std::vector<glm::vec4>* texcoord = NULL;

  glm::mat4 rotationM = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(1.0f, 0.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 20.0f - 0.5f*frame_, glm::vec3(0.0f, 1.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));

  cudaGLMapBufferObject((void**)&dptr, pbo_);
  octree_slam::rendering::rasterizeMesh(dptr, glm::vec2(width_, height_), rotationM, frame_, geometry.vbo, geometry.vbosize,
    newcbo_, 9, geometry.ibo, geometry.ibosize, geometry.nbo, geometry.nbosize, &texture, texcoord,
    camera.view, light, 0, false);
  cudaGLUnmapBufferObject(pbo_);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, displayImage_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

  frame_++;
}

void CUDARenderer::pixelPassthrough(const Color256* pixel_colors) {
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  uchar4 *dptr = NULL;

  cudaGLMapBufferObject((void**)&dptr, pbo_);
  writeColorToPBO(pixel_colors, dptr, width_*height_);
  cudaGLUnmapBufferObject(pbo_);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, displayImage_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

  frame_++;
}

void CUDARenderer::coneTraceSVO(const SVO& octree, const Camera& camera, const glm::vec3& light) {
  uchar4 *dptr = NULL;
  std::vector<glm::vec4>* texcoord = NULL;

  cudaGLMapBufferObject((void**)&dptr, pbo_);
  octree_slam::rendering::coneTraceSVO(dptr, glm::vec2(width_, height_), camera.fov, camera.view, octree);
  cudaGLUnmapBufferObject(pbo_);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, displayImage_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
}

} // namespace rendering

} // namespace octree_slam
