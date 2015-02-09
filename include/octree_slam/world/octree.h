#ifndef OCTREE_H_
#define OCTREE_H_

// OpenGL Dependencies
#include <glm/glm.hpp>

// Octree-SLAM Dependencies
#include <octree_slam/common_types.h>

namespace octree_slam {

namespace world {

class OctreeNode {

friend class Octree;

public:

  //Default Constructor does not allocate children
  OctreeNode();

  //Constructor that recursively creates children on CPU
  OctreeNode(const unsigned int morton_code, const int max_depth);

  ~OctreeNode();

private:

  //Recurses through children, pulling data from GPU to CPU
  void pullToCPU();

  //Recurses through children, pushing data from CPU to GPU
  void pushToGPU();

  //Gets the total number of allocated, non-zero children, recursively
  int totalChildren();

  //Recursively adds data to a stackless octree structure
  int addToLinearTree(int* octree, const int position, const int offset);

  //Recursively pulls data from a stackless octree structure into a set of child nodes
  void pullFromLinearTree(int* octree, const int position);

  //Allocates children on CPU
  void allocateChildren();

  //Expands the node by 1 size, maintaining its center
  void expand();

  //Flag whether the node is at max resolution
  bool is_max_depth_;

  //Flag whether the node's children have been initialized
  bool has_children_;

  //Flag whether the node's data is on the GPU
  bool on_gpu_;

  //Pointer to gpu data, if the node is GPU backed
  int* gpu_data_;

  //The number of children on the GPU
  int gpu_size_;

  //Child nodes if it is CPU backed
  OctreeNode* children_[8];

  //Data in the node if it is CPU backed
  int data_;

}; // class OctreeNode

class Octree {

public:

  //Constructor
  Octree(const float resolution, const glm::vec3& center, const float size);

  ~Octree();

  //Accessor function for getting a list of occupied voxels in the octree
  VoxelGrid occupiedVoxels() const;

  //Adds an observed point cloud
  void addCloud(const glm::vec3& origin, const glm::vec3* points, const Color256* colors, const int size);

  //Adds a voxelized mesh
  void addVoxelGrid(const VoxelGrid& grid);

private:

  //Expands the tree to a new size, keeping the current center
  void expandToSize(const float new_size);

  //Root node of the tree
  OctreeNode* root_;

  //Position of the root node
  glm::vec3 center_;

  //Size of the root node (half edge length)
  float size_;

}; // class Octree

} // namespace world

} // namespace octree_slam

#endif // OCTREE_H_
