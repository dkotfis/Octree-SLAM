
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
  //Determine how much space is needed and allocate it on the CPU
  int num_nodes = totalChildren();
  int* cpu_stackless = (int*) malloc(2*num_nodes*sizeof(int));

  //Fill in stackless octree data
  int offset = 8;
  for (int i = 0; i < 8; i++) {
    offset = children_[i]->addToLinearTree(cpu_stackless, i, offset);
  }

  //Copy it to the GPU
  cudaMalloc((void**)&gpu_data_, 2*num_nodes*sizeof(int));
  cudaMemcpy(gpu_data_, cpu_stackless, 2*num_nodes*sizeof(int), cudaMemcpyHostToDevice);

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
  //Don't do anything if there isn't at least 8 children worth of GPU data
  if (gpu_size_ < 8) {
    printf("[OctreeNode] Tried to pull data from GPU that was of insufficient size.");
    return;
  }

  //Copy data from GPU to CPU
  int* cpu_stackless = (int*) malloc(2*gpu_size_*sizeof(int));
  cudaMemcpy(cpu_stackless, gpu_data_, 2*gpu_size_*sizeof(int), cudaMemcpyDeviceToHost);
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

int OctreeNode::totalChildren() {
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

int OctreeNode::addToLinearTree(int* octree, const int position, const int offset) {
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

void OctreeNode::pullFromLinearTree(int* octree, const int position) {
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

Octree::Octree(const float resolution, const glm::vec3& center, const float size) : 
root_(new OctreeNode(1, (int) log((float) (size/resolution))/log(2.0f))), 
center_(center),
size_(size) {
}

Octree::~Octree() {
  //Delete the root, which recurisvely does the work
  delete root_;
}

VoxelGrid Octree::occupiedVoxels() const {
  VoxelGrid grid;

  return grid;
}

void Octree::addCloud(const glm::vec3& origin, const glm::vec3* points, const Color256* colors, const int size) {

}

void Octree::addVoxelGrid(const VoxelGrid& grid) {
  //TODO: Get the bounding box of the grid

  //TODO: Make sure the proper node is in GPU memory, and move it if it is not

  //TODO: Add voxels
}

void Octree::expandToSize(const float new_size) {
  //Determine how many additional layers will be needed
  int add_layers = log((float) (new_size / size_)) / log(2.0f);

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

} // namespace world

} // namespace octree_slam
