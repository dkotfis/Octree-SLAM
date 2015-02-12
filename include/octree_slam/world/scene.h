#ifndef SCENE_H_
#define SCENE_H_

#include <string>

// OpenGL Dependencies
#include <glm/glm.hpp>

#include <objUtil/objloader.h>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace world {

//Forward Declaration
class Octree;

class Scene {

public:

  Scene(const std::string& path_prefix);

  ~Scene();

  void loadObjFile(const std::string& filename);

  void loadBMP(const std::string& filename);

  void voxelizeMeshes(const bool octree = false);

  //Accessor method for meshes
  const std::vector<Mesh>& meshes() const { return meshes_; };

  //Accessor method for textures
  const std::vector<bmp_texture>& textures() const { return textures_; };

  //Accessor method for voxel grid
  const VoxelGrid& voxel_grid() const { return voxel_grid_; };

private:

  //Convenience utility for converting an obj to a Mesh struct
  Mesh objToMesh(obj* object);

  //The meshes in the scene
  std::vector<Mesh> meshes_;

  //The texture for each mesh in the scene
  std::vector<bmp_texture> textures_;

  //A voxel grid representing the scene
  VoxelGrid voxel_grid_;

  //A mesh for a single cube
  Mesh cube_;

  //The bounding box of the scene
  float bbox_[6];

  //An octree map representation of the scene
  Octree* tree_;

}; // class Scene

} // namespace world

} // namespace octree_slam

#endif // SCENE_H_
