#ifndef CUDA_RENDERER_H_
#define CUDA_RENDERER_H_

// OpenGL Dependencies
#include <glm/glm.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/scene_structs.h>

namespace octree_slam {

namespace rendering {

class CUDARenderer {

public:

  CUDARenderer(const bool voxelize, const std::string& path_prefix, const int width, const int height);

  ~CUDARenderer();

  void render(const Mesh& geometry, const bmp_texture& texture, const Camera& camera, const glm::vec3& light);

  void setPBO(const GLuint pbo) { pbo_ = pbo; use_other_pbo_ = true; };

private:

  static float newcbo_[9];

  const bool voxelized_;

  int width_;
  int height_;

  GLuint displayImage_;

  GLuint pbo_;
  bool use_other_pbo_;

  int frame_;

  static const GLuint positionLocation_ = 0;
  static const GLuint texcoordsLocation_ = 1;

}; // class CUDARenderer


} // namespace rendering

} // namespace octree_slam

#endif // CUDA_RENDERER_H_
