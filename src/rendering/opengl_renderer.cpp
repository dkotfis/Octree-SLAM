
#include <string>

// OpenGL Dependencies
#include <GL/glew.h>
#include <glslUtil/glslUtility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Octree-SLAM Dependency
#include <octree_slam/rendering/opengl_renderer.h>

namespace octree_slam {

namespace rendering {

float OpenGLRenderer::newcbo_[9] = { 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 };

OpenGLRenderer::OpenGLRenderer(const bool voxelize, const std::string& path_prefix) : voxelized_(voxelize) {

  glGenBuffers(3, buffers_);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  const char *attribLocations[] = { "v_position", "v_normal" };

  std::string vs, fs;
  if (voxelized_) {
    vs = path_prefix + "../shaders/voxels.vert";
    fs = path_prefix + "../shaders/voxels.frag";
  }
  else {
    vs = path_prefix + "../shaders/default.vert";
    fs = path_prefix + "../shaders/default.frag";
  }
  const char *vertShader = vs.c_str();
  const char *fragShader = fs.c_str();

  GLuint program = glslUtility::createProgram(attribLocations, 2, vertShader, fragShader);
  glUseProgram(program);

  mvp_location_ = glGetUniformLocation(program, "u_mvpMatrix");
  proj_location_ = glGetUniformLocation(program, "u_projMatrix");
  norm_location_ = glGetUniformLocation(program, "u_normMatrix");
  light_location_ = glGetUniformLocation(program, "u_light");

}

OpenGLRenderer::~OpenGLRenderer() {

}

void OpenGLRenderer::rasterize(const Mesh& geometry, const Camera& camera, const glm::vec3& light) {

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

} // namespace rendering

} // namespace octree_slam
