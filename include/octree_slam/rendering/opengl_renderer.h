#ifndef OPENGL_RENDERER_H_
#define OPENGL_RENDERER_H_

// OpenGL Dependencies
#include <glm/glm.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/scene_structs.h>

namespace octree_slam {

namespace rendering {

class OpenGLRenderer {

public:

  OpenGLRenderer(const bool voxelize, const std::string& path_prefix);

  ~OpenGLRenderer();

  void render(const Mesh& geometry, const Camera& camera, const glm::vec3& light);

private:

  static float newcbo_[9];

  const bool voxelized_;

  // Uniform locations for the shaders
  GLuint mvp_location_;
  GLuint proj_location_;
  GLuint norm_location_;
  GLuint light_location_;

  // VAO's
  GLuint buffers_[3];

}; // class OpenGLRenderer


} // namespace rendering

} // namespace octree_slam

#endif // OPENGL_RENDERER_H_
