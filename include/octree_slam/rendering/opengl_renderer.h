#ifndef OPENGL_RENDERER_H_
#define OPENGL_RENDERER_H_

// OpenGL Dependencies
#include <glm/glm.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace rendering {

class OpenGLRenderer {

public:

  OpenGLRenderer(const std::string &path_prefix);

  ~OpenGLRenderer();

  void rasterize(const Mesh &geometry, const Camera &camera, const glm::vec3 &light);

  void rasterizeVoxels(const VoxelGrid &geometry, const Camera &camera, const glm::vec3 &light);

  void renderPoints(const glm::vec3* positions, const Color256* colors, const int num, const Camera &camera);

private:

  static float newcbo_[9];

  //GLSL Programs
  GLuint default_program_;
  GLuint voxel_program_;
  GLuint points_program_;

  // VAO's
  GLuint buffers_[3];

  // Textures
  GLuint textures_[2];

}; // class OpenGLRenderer


} // namespace rendering

} // namespace octree_slam

#endif // OPENGL_RENDERER_H_
