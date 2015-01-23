
#include <string>

// OpenGL Dependencies
#include <GL/glew.h>
#include <glslUtil/glslUtility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

// CUDA Dependencies
#include <cuda_gl_interop.h>

// Octree-SLAM Dependency
#include <octree_slam/rendering/opengl_renderer.h>
#include <octree_slam/rendering/gl_interop_kernels.h>

namespace octree_slam {

namespace rendering {

float OpenGLRenderer::newcbo_[9] = { 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 };

OpenGLRenderer::OpenGLRenderer(const bool voxelize, const std::string& path_prefix) : voxelized_(voxelize) {

  glGenBuffers(3, buffers_);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  const char *attribLocations[] = { "v_position", "v_normal" };
  const char *attribLocations2[] = { "v_position", "v_normal", "v_color" };
  const char *attribLocations3[] = { "v_position", "v_color" };

  std::string vs, fs;
  
  vs = path_prefix + "../shaders/voxels.vert";
  fs = path_prefix + "../shaders/voxels.frag";

  const char *vertShader1 = vs.c_str();
  const char *fragShader1 = fs.c_str();

  voxel_program_ = glslUtility::createProgram(attribLocations, 2, vertShader1, fragShader1);

  vs = path_prefix + "../shaders/default.vert";
  fs = path_prefix + "../shaders/default.frag";

  const char *vertShader2 = vs.c_str();
  const char *fragShader2 = fs.c_str();

  default_program_ = glslUtility::createProgram(attribLocations2, 2, vertShader2, fragShader2);

  vs = path_prefix + "../shaders/points.vert";
  fs = path_prefix + "../shaders/points.frag";

  const char *vertShader3 = vs.c_str();
  const char *fragShader3 = fs.c_str();

  points_program_ = glslUtility::createProgram(attribLocations3, 2, vertShader3, fragShader3);

  glUseProgram(default_program_);

  mvp_location_ = glGetUniformLocation(default_program_, "u_mvpMatrix");
  proj_location_ = glGetUniformLocation(default_program_, "u_projMatrix");
  norm_location_ = glGetUniformLocation(default_program_, "u_normMatrix");
  light_location_ = glGetUniformLocation(default_program_, "u_light");
}

OpenGLRenderer::~OpenGLRenderer() {
}

void OpenGLRenderer::rasterize(const Mesh& geometry, const Camera& camera, const glm::vec3& light) {

  if (voxelized_) {
    glUseProgram(voxel_program_);
  } else {
    glUseProgram(default_program_);
  }

  //Send the MV, MVP, and Normal Matrices
  glUniformMatrix4fv(mvp_location_, 1, GL_FALSE, glm::value_ptr(camera.mvp));
  glUniformMatrix4fv(proj_location_, 1, GL_FALSE, glm::value_ptr(camera.projection));
  glm::mat3 norm_mat = glm::mat3(glm::transpose(glm::inverse(camera.model)));
  glUniformMatrix3fv(norm_location_, 1, GL_FALSE, glm::value_ptr(norm_mat));

  //Send the light position
  glUniform3fv(light_location_, 1, glm::value_ptr(light));

  // Send the VBO and NB0
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[0]);
  glBufferData(GL_ARRAY_BUFFER, geometry.vbosize*sizeof(float), geometry.vbo, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, buffers_[1]);
  glBufferData(GL_ARRAY_BUFFER, geometry.nbosize*sizeof(float), geometry.nbo, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  if (voxelized_) {
    glBindBuffer(GL_ARRAY_BUFFER, buffers_[2]);
    glBufferData(GL_ARRAY_BUFFER, geometry.cbosize*sizeof(float), geometry.cbo, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(2);
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLES, 0, geometry.vbosize);

}

void OpenGLRenderer::renderPoints(const Frame& frame, const Camera& camera) {
  //always use the point shaders to render points
  glUseProgram(points_program_);
  mvp_location_ = glGetUniformLocation(points_program_, "u_mvpMatrix");

  //Declare CUDA device pointers for it to use
  float3* dptr_pos;
  float3* dptr_col;

  //Setup position buffer
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[0]);
  glBufferData(GL_ARRAY_BUFFER, 3 * frame.width*frame.height*sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  //Setup color buffer
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[1]);
  glBufferData(GL_ARRAY_BUFFER, 3 * frame.width*frame.height*sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  //Register position and normal buffers with CUDA
  cudaGLRegisterBufferObject(buffers_[0]);
  cudaGLRegisterBufferObject(buffers_[1]);

  //Map buffers to CUDA
  cudaGLMapBufferObject((void**)&dptr_pos, buffers_[0]);
  cudaGLMapBufferObject((void**)&dptr_col, buffers_[1]);

  //Copy data to buffer with CUDA
  copyPointsToGL(frame.vertex, frame.color, dptr_pos, dptr_col, frame.width*frame.height);

  //Unmap buffers from CUDA
  cudaGLUnmapBufferObject(buffers_[0]);
  cudaGLUnmapBufferObject(buffers_[1]);

  //Unregister position and normal buffers with CUDA
  cudaGLUnregisterBufferObject(buffers_[0]);
  cudaGLUnregisterBufferObject(buffers_[1]);

  //Send the MVP Matrix
  glUniformMatrix4fv(mvp_location_, 1, GL_FALSE, glm::value_ptr(camera.mvp));

  //Draw
  glPointSize(1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArrays(GL_POINTS, 0, 3 * frame.width * frame.height);
}

} // namespace rendering

} // namespace octree_slam
