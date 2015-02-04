
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

OpenGLRenderer::OpenGLRenderer(const std::string& path_prefix) {

  glGenBuffers(3, buffers_);
  glGenTextures(2, textures_);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_BUFFER);

  glActiveTexture(GL_TEXTURE0);

  const char **attribLocations = NULL;
  const char *attribLocations2[] = { "v_position", "v_normal", "v_color" };
  const char *attribLocations3[] = { "v_position", "v_color" };

  std::string vs, fs;
  
  vs = path_prefix + "../shaders/voxels.vert";
  fs = path_prefix + "../shaders/voxels.frag";

  const char *vertShader1 = vs.c_str();
  const char *fragShader1 = fs.c_str();

  voxel_program_ = glslUtility::createProgram(attribLocations, 0, vertShader1, fragShader1);

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
}

OpenGLRenderer::~OpenGLRenderer() {
}

void OpenGLRenderer::rasterize(const Mesh& geometry, const Camera& camera, const glm::vec3& light) {

  glUseProgram(default_program_);

  GLuint mvp_location = glGetUniformLocation(default_program_, "u_mvpMatrix");
  GLuint norm_location = glGetUniformLocation(default_program_, "u_normMatrix");
  GLuint light_location = glGetUniformLocation(default_program_, "u_light");

  //Send the MVP, and Normal Matrices
  glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm::value_ptr(camera.mvp));
  glm::mat3 norm_mat = glm::mat3(glm::transpose(glm::inverse(camera.model)));
  glUniformMatrix3fv(norm_location, 1, GL_FALSE, glm::value_ptr(norm_mat));

  //Send the light position
  glUniform3fv(light_location, 1, glm::value_ptr(light));

  // Send the VBO and NB0
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[0]);
  glBufferData(GL_ARRAY_BUFFER, geometry.vbosize*sizeof(float), geometry.vbo, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, buffers_[1]);
  glBufferData(GL_ARRAY_BUFFER, geometry.nbosize*sizeof(float), geometry.nbo, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLES, 0, geometry.vbosize);

}

void OpenGLRenderer::rasterizeVoxels(const VoxelGrid& geometry, const Camera& camera, const glm::vec3& light) {
  //always use the voxel shaders to rasterize voxels with instancing
  glUseProgram(voxel_program_);
  GLuint mvp_location = glGetUniformLocation(voxel_program_, "u_mvpMatrix");
  GLuint norm_location = glGetUniformLocation(voxel_program_, "u_normMatrix");
  GLuint light_location = glGetUniformLocation(voxel_program_, "u_light");
  GLuint scale_location = glGetUniformLocation(voxel_program_, "u_scale");

  //Declare CUDA device pointers for it to use
  glm::vec4* dptr_centers;
  glm::vec4* dptr_colors;

  //Setup position buffer
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[0]);
  glBufferData(GL_ARRAY_BUFFER, 4*geometry.size*sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  //Setup color buffer
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * geometry.size*sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  //Register position and normal buffers with CUDA
  cudaGLRegisterBufferObject(buffers_[0]);
  cudaGLRegisterBufferObject(buffers_[1]);

  //Map buffers to CUDA
  cudaGLMapBufferObject((void**)&dptr_centers, buffers_[0]);
  cudaGLMapBufferObject((void**)&dptr_colors, buffers_[1]);

  //Copy data to buffer
  cudaMemcpy(dptr_centers, geometry.centers, 4*geometry.size*sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(dptr_colors, geometry.colors, 4*geometry.size*sizeof(float), cudaMemcpyDeviceToDevice);

  //Unmap buffers from CUDA
  cudaGLUnmapBufferObject(buffers_[0]);
  cudaGLUnmapBufferObject(buffers_[1]);

  //Unregister position and normal buffers with CUDA
  cudaGLUnregisterBufferObject(buffers_[0]);
  cudaGLUnregisterBufferObject(buffers_[1]);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_BUFFER, textures_[0]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, buffers_[0]);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_BUFFER, textures_[1]);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, buffers_[1]);

  //Send the MVP Matrix
  glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm::value_ptr(camera.mvp));
  glm::mat3 norm_mat = glm::mat3(glm::transpose(glm::inverse(camera.model)));
  glUniformMatrix3fv(norm_location, 1, GL_FALSE, glm::value_ptr(norm_mat));

  //Send the light position
  glUniform3fv(light_location, 1, glm::value_ptr(light));

  //Send the scale
  glUniform1f(scale_location, geometry.scale);

  //Draw
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArraysInstanced(GL_TRIANGLES, 0, 36, geometry.size);
}

void OpenGLRenderer::renderPoints(const glm::vec3* positions, const Color256* colors, const int num, const Camera &camera) {
  //always use the point shaders to render points
  glUseProgram(points_program_);
  GLuint mvp_location = glGetUniformLocation(points_program_, "u_mvpMatrix");

  //Declare CUDA device pointers for it to use
  float3* dptr_pos;
  float3* dptr_col;

  //Setup position buffer
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[0]);
  glBufferData(GL_ARRAY_BUFFER, 3 * num*sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  //Setup color buffer
  glBindBuffer(GL_ARRAY_BUFFER, buffers_[1]);
  glBufferData(GL_ARRAY_BUFFER, 3 * num*sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  //Register position and normal buffers with CUDA
  cudaGLRegisterBufferObject(buffers_[0]);
  cudaGLRegisterBufferObject(buffers_[1]);

  //Map buffers to CUDA
  cudaGLMapBufferObject((void**)&dptr_pos, buffers_[0]);
  cudaGLMapBufferObject((void**)&dptr_col, buffers_[1]);

  //Copy data to buffer with CUDA
  copyPointsToGL(positions, colors, dptr_pos, dptr_col, num);

  //Unmap buffers from CUDA
  cudaGLUnmapBufferObject(buffers_[0]);
  cudaGLUnmapBufferObject(buffers_[1]);

  //Unregister position and normal buffers with CUDA
  cudaGLUnregisterBufferObject(buffers_[0]);
  cudaGLUnregisterBufferObject(buffers_[1]);

  //Send the MVP Matrix
  glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm::value_ptr(camera.mvp));

  //Draw
  glPointSize(1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArrays(GL_POINTS, 0, 3 * num);
}

} // namespace rendering

} // namespace octree_slam
