
#include <cstdio>

// Octree-SLAM Dependency
#include <octree_slam/world/octree.h>
#include <octree_slam/world/svo/svo.h>

namespace octree_slam {

namespace world {

OctreeNode::OctreeNode() : 
is_max_depth_(false),
has_children_(false),
on_gpu_(false),
gpu_size_(0) {
}

OctreeNode::OctreeNode(const unsigned int morton_code, const int max_depth) :
has_children_(false),
on_gpu_(false),
gpu_size_(0) {
  //TODO: Determine is_max_depth_

  //TODO: If its not at max depth, set the has children flag and create children
}

OctreeNode::~OctreeNode() {
  //Free GPU data if its GPU backed
  if (on_gpu_) {
    cudaFree(gpu_data_);

  //Free CPU data if it has children on the CPU
  } else if (has_children_) {
    for (int i = 0; i < 8; i++) {
      delete children_[i];
    }
  }
}

void OctreeNode::pushToGPU() {
  //Don't need to repush if its on the GPU already
  if (on_gpu_) {
    return;
  }

  //Can't push if it hasn't been allocated
  if (!has_children_) {
    on_gpu_ = true;
    return;
  }

  //Determine how much space is needed and allocate it on the CPU
  int num_nodes = totalChildren();
  unsigned int* cpu_stackless = (unsigned int*) malloc(2*num_nodes*sizeof(unsigned int));

  //Fill in stackless octree data
  int offset = 8;
  for (int i = 0; i < 8; i++) {
    offset = children_[i]->addToLinearTree(cpu_stackless, i, offset);
  }

  //Copy it to the GPU
  cudaMalloc((void**)&gpu_data_, 2*num_nodes*sizeof(unsigned int));
  cudaMemcpy(gpu_data_, cpu_stackless, 2*num_nodes*sizeof(unsigned int), cudaMemcpyHostToDevice);

  //Free the CPU copy
  free(cpu_stackless);

  //Delete the stack-based copy
  for (int i = 0; i < 8; i++) {
    delete children_[i];
  }

  //Update node params
  gpu_size_ = num_nodes;
  on_gpu_ = true;
  has_children_ = false;
}

void OctreeNode::pullToCPU() {
  //Don't do anything if its already on the CPU
  if (!on_gpu_) {
    return;
  }

  //Don't do anything if there isn't at least 8 children worth of GPU data
  if (gpu_size_ < 8) {
    printf("[OctreeNode] Tried to pull data from GPU that was of insufficient size.");
    return;
  }

  //Copy data from GPU to CPU
  unsigned int* cpu_stackless = (unsigned int*) malloc(2*gpu_size_*sizeof(unsigned int));
  cudaMemcpy(cpu_stackless, gpu_data_, 2*gpu_size_*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(gpu_data_);

  //Allocate children and fill them
  allocateChildren();
  for (int i = 0; i < 8; i++) {
    children_[i]->pullFromLinearTree(cpu_stackless, i);
  }

  //Free the CPU copy
  free(cpu_stackless);

  //Update node params
  gpu_size_ = 0;
  on_gpu_ = false;
  has_children_ = true;
}

int OctreeNode::totalChildren() const {
  //Fill in for this node
  int total = (data_ != 0) ? 1 : 0;

  //Recursively fill CPU children
  if (has_children_) {
    for (int i = 0; i < 8; i++) {
      total += children_[i]->totalChildren();
    }
  }

  //Add GPU children
  total += gpu_size_;

  return total;
}

int OctreeNode::addToLinearTree(unsigned int* octree, const int position, const int offset) {
  int new_offset = offset;

  //Add this node's data
  octree[2*position] = (octree[2*position] & 0xC0000000) | (offset & 0x3FFFFFFF);
  octree[2*position + 1] = data_;

  if (has_children_) {
    //Recurse children
    for (int i = 0; i < 8; i++) {
      new_offset = children_[i]->addToLinearTree(octree, offset + i, new_offset + 8);
    }
    new_offset += 8;
  } else if (on_gpu_) {
    //Copy GPU data
    cudaMemcpy(octree+offset, gpu_data_, 2*gpu_size_*sizeof(int), cudaMemcpyDeviceToDevice);
    new_offset += gpu_size_;
  } else {
    octree[2 * position] = 0;
  }

  return new_offset;
}

void OctreeNode::pullFromLinearTree(unsigned int* octree, const int position) {
  //If the has child flag is set, allocate children and recurse them
  if (octree[2 * position] & 0x40000000) {
    allocateChildren();

    //Get the offset pointer to the children (lower 30 bits)
    int offset = octree[2*position] & 0x3FFFFFFF;;

    for (int i = 0; i < 8; i++) {
      children_[i]->pullFromLinearTree(octree, offset+i);
    }
  }

  //Get this node's value
  data_ = octree[2*position + 1];
}

void OctreeNode::allocateChildren() {
  //Don't reallocate if children are already there
  if (has_children_) {
    printf("[OctreeNode] Requested allocation of children on a node that has already been allocated.");
    return;
  }

  for (int i = 0; i < 8; i++) {
    children_[i] = new OctreeNode();
  }
  has_children_ = true;
}

void OctreeNode::expand() {
  //Don't do anything if this is at max_depth or has no allocated children
  if (is_max_depth_ || (!has_children_ && !on_gpu_)) {
    printf("[OctreeNode] Cannot expand a node that is at max_depth or has no children.");
    return;
  }

  //TODO: Handle the case where the node is GPU backed
  if (on_gpu_) {
    printf("[OctreeNode] Cannot currently expand a node that is on the GPU.");
    return;
  }

  //Create new children, point to existing children, and update
  for (int i = 0; i < 8; i++) {
    OctreeNode* new_child = new OctreeNode();
    new_child->allocateChildren();
    delete new_child->children_[svo::oppositeNode(i)];
    new_child->children_[svo::oppositeNode(i)] = children_[i];
    children_[i] = new_child;
  }

}

OctreeNode* OctreeNode::getNodeContainingBoundingBox(const BoundingBox& bbox, BoundingBox& this_bbox, int& current_depth, glm::vec3& result_center) {
  //Use this node if it isn't allocated yet or its all on the GPU
  if (!has_children_ || on_gpu_) {
    return this;
  }

  //Compute the center of this node
  glm::vec3 center = (this_bbox.bbox1 + this_bbox.bbox0) / 2.0f;

  //Determine whether the full bounding box fits in one of the children
  bool all_together = true;
  bool x_plus, y_plus, z_plus;
  if (x_plus = (bbox.bbox1.x > center.x) != (bbox.bbox0.x > center.x)) {
    all_together = false;
  }
  else if (y_plus = (bbox.bbox1.y > center.y) != (bbox.bbox0.y > center.y)) {
    all_together = false;
  }
  else if (z_plus = (bbox.bbox1.z > center.z) != (bbox.bbox0.z > center.z)) {
    all_together = false;
  }

  //If the corners are in different children, this is the node we want
  if (!all_together) {
    result_center = center;
    return this;
  }
  //Else continue to search the child that contains them
  else {
    //Construct the bounding box for the child
    this_bbox.bbox0.x = x_plus ? center.x : this_bbox.bbox0.x;
    this_bbox.bbox1.x = x_plus ? this_bbox.bbox1.x : center.x;
    this_bbox.bbox0.y = y_plus ? center.y : this_bbox.bbox0.y;
    this_bbox.bbox1.y = y_plus ? this_bbox.bbox1.y : center.y;
    this_bbox.bbox0.z = z_plus ? center.z : this_bbox.bbox0.z;
    this_bbox.bbox1.z = z_plus ? this_bbox.bbox1.z : center.z;

    //Recurse into that child
    int child_idx = x_plus + 2 * y_plus + 4 * z_plus;
    return children_[child_idx]->getNodeContainingBoundingBox(bbox, this_bbox, ++current_depth, result_center);
  }
}

Octree::Octree(const float resolution, const glm::vec3& center, const float size) : 
root_(new OctreeNode(1, (int) log((float) (size/resolution))/log(2.0f) + 1)), 
center_(center),
size_(size),
resolution_(resolution) {
}

Octree::~Octree() {
  //Delete the root, which recurisvely does the work
  delete root_;
}

VoxelGrid Octree::occupiedVoxels() const {
  VoxelGrid grid;

  return grid;
}

void Octree::addCloud(const glm::vec3& origin, const glm::vec3* points, const Color256* colors, const int size, const BoundingBox& bbox) {
  //TODO: Almost all of this is reused from addVoxelGrid. Refactor the API

  //Compute the bounding box of the root
  BoundingBox root_box;
  root_box.bbox0 = center_ - glm::vec3(size_, size_, size_);
  root_box.bbox1 = center_ + glm::vec3(size_, size_, size_);

  //Get the node for this bounding box
  glm::vec3 node_cent = center_;
  int node_depth = 0;
  OctreeNode* subtree = root_->getNodeContainingBoundingBox(bbox, root_box, node_depth, node_cent);

  //Calculate the edge length and depth of this node
  float edge_length = size_ / pow(2.0f, (float)node_depth);
  int max_depth = ceil(log((float)(edge_length / resolution_)) / log(2.0f));

  //Make sure the subtree is in GPU memory
  subtree->pushToGPU();

  //Add points
  svo::svoFromPointCloud(points, colors, size, max_depth, subtree->gpu_data_, subtree->gpu_size_, node_cent, edge_length);
}

void Octree::addVoxelGrid(const VoxelGrid& grid) {
  //Compute the bounding box of the root
  BoundingBox root_box;
  root_box.bbox0 = center_ - glm::vec3(size_, size_, size_);
  root_box.bbox1 = center_ + glm::vec3(size_, size_, size_);

  //Get the node for this bounding box
  glm::vec3 node_cent = center_;
  int node_depth = 0;
  OctreeNode* subtree = root_->getNodeContainingBoundingBox(grid.bbox, root_box, node_depth, node_cent);

  //Calculate the edge length and depth of this node
  float edge_length = size_ / pow(2.0f, (float)node_depth);
  int max_depth = ceil(log((float)(edge_length / resolution_)) / log(2.0f));

  //Make sure the subtree is in GPU memory
  subtree->pushToGPU();

  //Add voxels to it
  svo::svoFromVoxelGrid(grid, max_depth, subtree->gpu_data_, subtree->gpu_size_, node_cent, edge_length); 
}

void Octree::extractVoxelGrid(VoxelGrid& grid) {
  //TODO: Almost all of this is reused from addVoxelGrid. Refactor the API

  //Compute the bounding box of the root
  BoundingBox root_box;
  root_box.bbox0 = center_ - glm::vec3(size_, size_, size_);
  root_box.bbox1 = center_ + glm::vec3(size_, size_, size_);

  //Get the node for this bounding box
  glm::vec3 node_cent = center_;
  int node_depth = 0;
  OctreeNode* subtree = root_->getNodeContainingBoundingBox(grid.bbox, root_box, node_depth, node_cent);

  //Calculate the edge length and depth of this node
  float edge_length = size_ / pow(2.0f, (float)node_depth);
  int max_depth = ceil(log((float)(edge_length / grid.scale)) / log(2.0f));

  //Make sure the subtree is in GPU memory
  subtree->pushToGPU();

  //Pull voxel data from the octree pool
  svo::extractVoxelGridFromSVO(subtree->gpu_data_, subtree->gpu_size_, max_depth, node_cent, edge_length, grid);
}

void Octree::expandBySize(const float add_size) {
  //Determine how many additional layers will be needed
  int add_layers = log(ceil((size_ + add_size) / size_)) / log(2.0f);

  //Don't continue if we're not adding at least 1 layer
  if (add_layers < 1) {
    return;
  }

  //Expand the root node the needed number of times
  for (int i = 1; i < add_layers; i++) {
    root_->expand();
  }

  //Set the new size
  size_ = pow(2.0f, (float)add_layers) * size_;
}

BoundingBox Octree::boundingBox() const {
  BoundingBox box;
  box.bbox0 = center_ - glm::vec3(size_, size_, size_);
  box.bbox1 = center_ + glm::vec3(size_, size_, size_);
  return box;
}

} // namespace world

} // namespace octree_slam
