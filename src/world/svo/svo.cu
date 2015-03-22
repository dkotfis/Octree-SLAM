
#include <stack>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

// Octree-SLAM Dependencies
#include <octree_slam/timing_utils.h>
#include <octree_slam/world/svo/svo.h>

namespace octree_slam {

namespace svo {

texture<float, 3> brick_tex;
surface<void, 3> brick_surf;

__host__ void initOctree(unsigned int* &octree) {
  int oct[16];
  for (int i = 0; i < 16; i++) {
    oct[i] = 0;
  }
  cudaMalloc((void**)&octree, 16*sizeof(unsigned int));
  cudaMemcpy(octree, oct, 16*sizeof(unsigned int), cudaMemcpyHostToDevice);
}

template <class T>
__device__ int computeKey(const T& point, glm::vec3 center, const int tree_depth, float edge_length) {
  //TODO: This will break if tree_depth > 10 (would require >= 33 bits)

  //Check for invalid
  if (!isfinite(point.x) || !isfinite(point.z) || !isfinite(point.z)) {
    return 1;
  }

  //Initialize the output value
  int morton = 0;

  for (int i = 0; i < tree_depth; i++) {
    //Determine which octant the point lies in
    bool x = point.x > center.x;
    bool y = point.y > center.y;
    bool z = point.z > center.z;

    //Update the code
    morton += ((x + 2*y + 4*z) << 3*i);

    //Update the edge length
    edge_length /= 2.0f;

    //Update the center
    center.x += edge_length * (x ? 1 : -1);
    center.y += edge_length * (y ? 1 : -1);
    center.z += edge_length * (z ? 1 : -1);
  }

  //Add a leading 1 to specify the depth
  morton += (1 << tree_depth*3);

  return morton;
}

template <class T>
__global__ void computeKeys(const T* voxels, const int numVoxels, const int max_depth, const glm::vec3 center, float edge_length, int* keys) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numVoxels) {
    return;
  }

  //Compute the full morton code
  const int morton = computeKey(voxels[index], center, max_depth, edge_length);

  //Fill in the key for the output
  keys[index] = morton;
}

__global__ void splitKeys(int* keys, const int numKeys, unsigned int* octree, int* left, int* right) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  int l_key = keys[index];
  int r_key = -1;
  int temp_key = 0;

  int node_idx = 0;
  int depth = 0;

  //Determine the existing depth from the current octree data
  while (l_key >= 15) {
    //Get the child number from the bottom three bits of the key
    temp_key += (l_key & 0x7) << 3*depth;
    node_idx += (l_key & 0x7);

    //Check the flag
    if (!(octree[2 * node_idx] & 0x40000000)) {
      r_key = (1 << 3*(depth+1)) + temp_key;
      break;
    }

    //The lowest 30 bits are the address of the child nodes
    node_idx = octree[2 * node_idx] & 0x3FFFFFFF;

    //Update the key
    l_key = l_key >> 3;

    depth++;
  }

  //Fill in the results
  left[index] = l_key;
  right[index] = r_key;
}

__device__ int depthFromKey(int key) {
  const int bval[] =
  { 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };

  int r = 0;
  if (key & 0x7FFF0000) { r += 16 / 1; key >>= 16 / 1; }
  if (key & 0x0000FF00) { r += 16 / 2; key >>= 16 / 2; }
  if (key & 0x000000F0) { r += 16 / 4; key >>= 16 / 4; }

  return (r + bval[key] - 1) / 3;
}

__global__ void leftToRightShift(int* left, int* right, const int numVoxels) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numVoxels) {
    return;
  }

  //Not valid if left is empty or if right is already invalid
  if (right[index] == -1 || left[index] == 1) {
    right[index] = -1;
    return;
  }

  //Get moving bits from left
  int moved_bits = left[index] & 0x7;

  //Remove them from left
  left[index] = left[index] >> 3;

  //If this leaves left empty (only the leading 1) then right is no longer valid
  if (left[index] == 1) {
    right[index] = -1;
    return;
  }

  //Get a copy of the right key
  int r_key = right[index];

  //Compute the depth of the current right key
  int depth = depthFromKey(r_key);

  //Remove the leading one at the old depth, and bump it up
  r_key -= (1 << 3 * depth);
  r_key += (1 << 3 * (depth+1));

  //Add the moved bits
  right[index] = r_key + (moved_bits << 3*depth);
}

struct negative {
  __host__ __device__ bool operator() (const int x) {
    return (x < 0);
  }
};

__host__ int prepassCheckResize(int* keys, const int numVoxels, const int max_depth, unsigned int* octree, int** &d_codes, int* &code_sizes) {
  int num_split_nodes = 0;

  //Allocate left and right morton code data
  int* d_left;
  cudaMalloc((void**)&d_left, numVoxels*sizeof(int));
  int* d_right;
  cudaMalloc((void**)&d_right, numVoxels*sizeof(int));
  int* temp_right;
  cudaMalloc((void**)&temp_right, numVoxels*sizeof(int));
  thrust::device_ptr<int> t_right = thrust::device_pointer_cast<int>(temp_right);

  //Split the set of existing codes and new codes (right/left)
  splitKeys<<<ceil(float(numVoxels)/128.0f), 128>>>(keys, numVoxels, octree, d_left, d_right);
  cudaDeviceSynchronize();

  //Allocate memory for output code data based on max_depth
  d_codes = (int**)malloc(max_depth*sizeof(int*));
  code_sizes = (int*)malloc(max_depth*sizeof(int));

  //Loop over passes
  for (int i = 0; i < max_depth; i++) {
    //Get a copy of the right codes so we can modify
    cudaMemcpy(thrust::raw_pointer_cast(t_right), d_right, numVoxels*sizeof(int), cudaMemcpyDeviceToDevice);

    //Get the valid codes
    int size = thrust::remove_if(t_right, t_right + numVoxels, negative()) - t_right;

    //If there are no valid codes, we're done
    if (size == 0) {
      for (int j = i; j < max_depth; j++) {
        code_sizes[j] = 0;
      }
      break;
    }

    //Get the unique codes
    thrust::sort(t_right, t_right + size);
    size = thrust::unique(t_right, t_right + size) - t_right;

    //Allocate output and copy the data into it
    code_sizes[i] = size;
    cudaMalloc((void**)&(d_codes[i]), size*sizeof(int));
    cudaMemcpy(d_codes[i], thrust::raw_pointer_cast(t_right), size*sizeof(int), cudaMemcpyDeviceToDevice);

    //Update the total number of nodes that are being split
    num_split_nodes += size;

    //Move one morton code from each right to left for the next depth
    leftToRightShift<<<ceil((float)numVoxels/128.0f), 128>>>(d_left, d_right, numVoxels);
  }

  //Cleanup
  cudaFree(d_left);
  cudaFree(d_right);
  cudaFree(temp_right);

  return num_split_nodes;
}

__global__ void splitNodes(const int* keys, int numKeys, unsigned int* octree, int num_nodes) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  int key = keys[index];

  //Don't do anything if its an empty key
  if (key == 1) {
    return;
  }

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the bottom three bits of the key
    node_idx = child_idx + (key & 0x7);
    key = key >> 3;

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  //Get a new node tile
  int newNode = num_nodes + 8 * index;

  //Point this node at the new tile, and flag it
  octree[2 * node_idx] = (1 << 30) +  (newNode & 0x3FFFFFFF);

  //Initialize new child nodes to 0's
  for (int off = 0; off < 8; off++) {
    octree[2 * (newNode + off)] = 0;
    octree[2 * (newNode + off) + 1] = 127 << 24;
  }
}

__host__ void expandTreeAtKeys(int** d_keys, int* numKeys, const int depth, unsigned int* octree, int& num_nodes) {
  for (size_t i = 0; i < depth; i++) {
    if (numKeys[i] == 0) {
      break;
    }

    splitNodes<<<ceil((float)numKeys[i]/128.0f), 128>>>(d_keys[i], numKeys[i], octree, num_nodes);

    num_nodes += 8 * numKeys[i];
    cudaDeviceSynchronize();
  }
}

__global__ void fillNodes(const int* keys, int numKeys, const glm::vec4* values, unsigned int* octree) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  int key = keys[index];

  //Check for invalid key
  if (key == 1) {
    return;
  }

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the bottom three bits of the morton code
    node_idx = child_idx + (key & 0x7);
    key = key >> 3;

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  glm::vec4 new_value = values[index] * 256.0f;
  unsigned int current_value = octree[2 * node_idx + 1];

  int current_alpha = current_value >> 24;
  int current_r = current_value & 0xFF;
  int current_g = (current_value >> 8) & 0xFF;
  int current_b = (current_value >> 16) & 0xFF;

  //Implement a pseudo low-pass filter with laplace smoothing
  float f1 = 1 - ((float)current_alpha / 256.0f);
  float f2 = (float)current_alpha / 256.0f;
  new_value.r = new_value.r * f1 + current_r * f2;
  new_value.g = new_value.g * f1 + current_g * f2;
  new_value.b = new_value.b * f1 + current_b * f2;
  octree[2 * node_idx + 1] = ((int)new_value.r) + ((int)new_value.g << 8) + ((int)new_value.b << 16) + (min(255, current_alpha + 2) << 24);
}

__global__ void fillNodes(const int* keys, int numKeys, const Color256* values, unsigned int* octree) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  int key = keys[index];

  //Check for invalid key
  if (key == 1) {
    return;
  }

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the bottom three bits of the morton code
    node_idx = child_idx + (key & 0x7);

    if (!octree[2 * node_idx] & 0x40000000) {
      return;
    }

    key = key >> 3;

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  Color256 new_value = values[index];
  unsigned int current_value = octree[2 * node_idx + 1];

  Color256 current;
  short current_alpha = current_value >> 24;
  current.r = current_value & 0xFF;
  current.g = (current_value >> 8) & 0xFF;
  current.b = (current_value >> 16) & 0xFF;

  //Implement a pseudo low-pass filter with laplace smoothing
  float f1 = (1 - ((float)current_alpha/256.0f));
  float f2 = (float)current_alpha / 256.0f;
  new_value.r = new_value.r * f1 + current.r * f2;
  new_value.g = new_value.g * f1 + current.g * f2;
  new_value.b = new_value.b * f1 + current.b * f2;
  octree[2 * node_idx + 1] = ((int)new_value.r) + ((int)new_value.g << 8) + ((int)new_value.b << 16) + (min(255, current_alpha + 2) << 24);
}

__global__ void averageChildren(int* keys, int numKeys, unsigned int* octree) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  int key = keys[index];

  ///Get the depth of the key
  int depth = depthFromKey(key);

  //Remove the max depth level from the key
  int mask = ~(0xFFFFFFFF & (0x3F << 3*(depth-1)));
  key = (key & mask) + (1 << 3*(depth-1));

  //Fill in back into global memory this way
  keys[index] = key;

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the bottom three bits of the morton code
    node_idx = child_idx + (key & 0x7);
    key = key >> 3;

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  //Loop through children values and average them
  glm::vec4 val = glm::vec4(0.0);
  int num_occ = 0;
  for (int i = 0; i < 8; i++) {
    int child_val = octree[2*(child_idx+i) + 1];
    if ((child_val >> 24) & 0xFF == 0) {
      //Don't count in the average if its not occupied
      continue;
    }
    val.r += (float) (child_val & 0xFF);
    val.g += (float) ((child_val >> 8) & 0xFF);
    val.b += (float) ((child_val >> 16) & 0xFF);
    //Assign the max albedo (avoids diluting it)
    val.a = max(val.a,(float) ((child_val >> 24) & 0xFF));
    num_occ++;
  }

  //Average the color values
  if (num_occ > 0) {
    val.r = val.r / (float)num_occ;
    val.g = val.g / (float)num_occ;
    val.b = val.b / (float)num_occ;
  }

  //Assign value of this node to the average
  octree[(2 * node_idx) + 1] = (int)val.r + ((int)val.g << 8) + ((int)val.b << 16) + ((int)val.a << 24);

}

struct depth_is_zero {
  __device__ bool operator() (const int key) {
    return (depthFromKey(key) == 0);
  }
};


__host__ void mipmapNodes(int* keys, int numKeys, unsigned int* octree) {

  //Get a thrust pointer for the keys
  thrust::device_ptr<int> t_keys = thrust::device_pointer_cast<int>(keys);

  //Check if any keys still have children
  while ((numKeys = thrust::remove_if(t_keys, t_keys + numKeys, depth_is_zero()) - t_keys) > 0) {
    //thrust::sort(t_keys, t_keys + numKeys);
    //numKeys = thrust::unique(t_keys, t_keys + numKeys) - t_keys;

    //Average the children at the given set of keys
    averageChildren<<<ceil((float)numKeys / 32.0f), 32>>>(keys, numKeys, octree);
    cudaDeviceSynchronize();
  }

}

/*
__global__ void mipmapBricks(int* octree, int poolSize, int startNode, cudaArray* bricks, float* numBricks) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= poolSize) {
    return;
  }

  int node = octree[2 * (index + startNode)];

  //Don't do anything if this node has no children
  if (!(node & 0x40000000)) {
    return;
  }

  //Get a new brick
  float newBrick = atomicAdd(numBricks, 3.0f);

  //Assign the brick to the node
  octree[(2 * (index + startNode)) + 1] = __float_as_int(newBrick);

  //TODO: Get all necessary neighbors

  //TODO: Fill the values into the brick in texture memory
  float val = tex3D(brick_tex, 1.1f, 1.1f, 1.1f);
  surf3Dwrite(5.0f, brick_surf, 1, 1, 1, cudaBoundaryModeClamp);
}
*/

__global__ void getOccupiedChildren(const unsigned int* octree, const int* parents, const int num_parents, int* children) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= num_parents) {
    return;
  }

  //Get the key for the parent
  const int key = parents[index];
  int temp_key = key;

  //Flag whether the node has any children
  int has_children = true;

  //Get the pointer to the children
  int pointer = 0;
  while (temp_key != 1) {
    //Get the next child
    pointer += temp_key & 0x7;
    has_children = octree[2 * pointer] & 0x40000000;
    pointer = octree[2 * pointer] & 0x3FFFFFFF;

    //Update the key
    temp_key = temp_key >> 3;
  }

  //Loop through the children, and if they are occupied fill its key into the output
  for (int i = 0; i < 8; i++) {
    int child_val = -1;

    if (has_children) {
      unsigned int val2 = octree[2 * (pointer + i) + 1];
      if (((val2 >> 24) & 0xFF) > 127) { //TODO: Should we threshold "occupied" at something other than 0?
        temp_key = key;

        //Compute the depth of the current key
        int depth = depthFromKey(temp_key);

        //Remove the leading one at the old depth, and bump it up
        temp_key -= (1 << 3 * depth);
        temp_key += (1 << 3 * (depth + 1));

        //Add the moved bits
        child_val = temp_key + (i << 3 * depth);
      }
    }

    children[8 * index + i] = child_val;
  }
}

__global__ void voxelGridFromKeys(unsigned int* octree, int* keys, int num_voxels, glm::vec3 center, float edge_length, glm::vec4* centers, glm::vec4* colors) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  //Don't do anything if out of bounds
  if (idx >= num_voxels) {
    return;
  }

  int key = keys[idx];

  //Get the pointer to the voxel
  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the next child
    int pos = key & 0x7;
    node_idx = child_idx + pos;
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;

    //Update the key
    key = key >> 3;

    //Decode the value into xyz
    int x = pos & 0x1;
    int y = pos & 0x2;
    int z = pos & 0x4;

    //Half the edge length to use it for the update
    edge_length /= 2.0f;

    //Update the center
    center.x += edge_length * (x ? 1 : -1);
    center.y += edge_length * (y ? 1 : -1);
    center.z += edge_length * (z ? 1 : -1);
  }

  unsigned int val = octree[2 * node_idx + 1];

  //Fill in the voxel
  centers[idx] = glm::vec4(center.x, center.y, center.z, 1.0f);
  colors[idx].r = ((float)(val & 0xFF) / 255.0f);
  colors[idx].g = ((float)((val >> 8) & 0xFF) / 255.0f);
  colors[idx].b = ((float)((val >> 16) & 0xFF) / 255.0f);
  colors[idx].a = ((float)((val >> 24) & 0xFF) / 255.0f);

}

extern "C" void svoFromVoxelGrid(const VoxelGrid& grid, const int max_depth, unsigned int* &octree, int& octree_size, glm::vec3 octree_center, const float edge_length, cudaArray* d_bricks) {

  //Initialize the octree with a base set of empty nodes if its empty
  if (octree_size == 0) {
    initOctree(octree);
    octree_size = 8;
  }

  //Allocate space for octree keys for each input
  int* d_keys;
  cudaMalloc((void**)&d_keys, grid.size*sizeof(int));

  //Compute Keys
  computeKeys<glm::vec4><<<ceil((float)grid.size/256.0f), 256>>>(grid.centers, grid.size, max_depth, octree_center, edge_length, d_keys);
  cudaDeviceSynchronize();

  //Determine how many new nodes are needed in the octree, and the keys to the nodes that need split in each pass
  int** d_codes = NULL;
  int* code_sizes = NULL;
  int new_nodes = prepassCheckResize(d_keys, grid.size, max_depth, octree, d_codes, code_sizes);

  //Create a new octree with an updated size and copy over the old data. Free up the old copy when it is no longer needed
  unsigned int* new_octree;
  cudaMalloc((void**)&new_octree, 2 * (octree_size + 8*new_nodes) * sizeof(unsigned int));
  cudaMemcpy(new_octree, octree, 2 * octree_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  cudaFree(octree);
  octree = new_octree;

  //Expand the tree now that the space has been allocated
  expandTreeAtKeys(d_codes, code_sizes, max_depth, octree, octree_size);

  //Free up the codes now that we no longer need them
  for (size_t i = 0; i < max_depth; i++) {
    if (code_sizes[i] > 0) {
      cudaFree(d_codes[i]);
    }
  }
  free(d_codes);
  free(code_sizes);

  //Write voxel values into the lowest level of the svo
  fillNodes<<<ceil((float)grid.size / 256.0f), 256>>>(d_keys, grid.size, grid.colors, octree);
  cudaDeviceSynchronize();
  //TODO: Handle duplicate keys

  //Mip-mapping (currently only without use of the brick pool)
  mipmapNodes(d_keys, grid.size, octree);
  cudaDeviceSynchronize();

  //Free up the keys since they are no longer needed
  cudaFree(d_keys);

}

extern "C" void svoFromPointCloud(const glm::vec3* points, const Color256* colors, const int size, const int max_depth, unsigned int* &octree, int& octree_size, glm::vec3 octree_center, const float edge_length, cudaArray* d_bricks) {
  //TODO: This duplicates alot from the VoxelGrid function. Refactor the API to be more efficient

  //Initialize the octree with a base set of empty nodes if its empty
  if (octree_size == 0) {
    initOctree(octree);
    octree_size = 8;
  }
  //Allocate space for octree keys for each input
  int* d_keys;
  cudaMalloc((void**)&d_keys, size*sizeof(int));

  //Compute Keys
  computeKeys<glm::vec3><<<ceil((float)size / 256.0f), 256>>>(points, size, max_depth, octree_center, edge_length, d_keys);
  cudaDeviceSynchronize();

  //Determine how many new nodes are needed in the octree, and the keys to the nodes that need split in each pass
  int** d_codes = NULL;
  int* code_sizes = NULL;
  int new_nodes = prepassCheckResize(d_keys, size, max_depth, octree, d_codes, code_sizes);

  //Create a new octree with an updated size and copy over the old data. Free up the old copy when it is no longer needed
  unsigned int* new_octree;
  cudaMalloc((void**)&new_octree, 2 * (octree_size + 8 * new_nodes) * sizeof(unsigned int));
  cudaMemcpy(new_octree, octree, 2 * octree_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  cudaFree(octree);
  octree = new_octree;

  //Expand the tree now that the space has been allocated
  expandTreeAtKeys(d_codes, code_sizes, max_depth, octree, octree_size);
  //Free up the codes now that we no longer need them
  for (size_t i = 0; i < max_depth; i++) {
    if (code_sizes[i] > 0) {
      cudaFree(d_codes[i]);
    }
  }
  free(d_codes);
  free(code_sizes);

  //Write voxel values into the lowest level of the svo
  fillNodes<<<ceil((float)size / 256.0f), 256>>>(d_keys, size, colors, octree);
  cudaDeviceSynchronize();
  //TODO: Handle duplicate keys

  //Mip-mapping (currently only without use of the brick pool)
  mipmapNodes(d_keys, size, octree);
  cudaDeviceSynchronize();

  //Free up the keys since they are no longer needed
  cudaFree(d_keys);
}


extern "C" void extractVoxelGridFromSVO(unsigned int* &octree, int& octree_size, const int max_depth, const glm::vec3 center, float edge_length, VoxelGrid& grid) {

  //Loop through each pass until max_depth, and determine the number of nodes at the highest resolution, along with morton codes for them
  int num_voxels = 1;

  //Initialize a node list with empty key (only a leading 1) for the first set of children, and copy to GPU
  int initial_nodes[1] = {1};
  int* node_list;
  cudaMalloc((void**)&node_list, sizeof(int));
  cudaMemcpy(node_list, initial_nodes, sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < max_depth; i++) {
    //Allocate space for this pass based on the number of keys (x8)
    int* new_nodes;
    cudaMalloc((void**)&new_nodes, 8*num_voxels*sizeof(int));

    //Run kernel on all of the keys (x8)
    getOccupiedChildren<<<ceil((float)num_voxels/256.0f), 256>>>(octree, node_list, num_voxels, new_nodes);
    cudaDeviceSynchronize();

    //Thrust remove-if to get the set of keys for the next pass
    {
      thrust::device_ptr<int> t_nodes = thrust::device_pointer_cast<int>(new_nodes);
      num_voxels = thrust::remove_if(t_nodes, t_nodes + 8*num_voxels, negative()) - t_nodes;
    }

    //Free up memory for the previous set of keys
    cudaFree(node_list);
    node_list = new_nodes;
  }

  //Allocate the voxel grid
  grid.size = num_voxels;
  cudaMalloc((void**)&grid.centers, num_voxels*sizeof(glm::vec4));
  cudaMalloc((void**)&grid.colors, num_voxels*sizeof(glm::vec4));

  //Extract the data into the grid
  voxelGridFromKeys<<<ceil((float)num_voxels / 256.0f), 256>>>(octree, node_list, num_voxels, center, edge_length, grid.centers, grid.colors);
  cudaDeviceSynchronize();

  //Free up memory
  cudaFree(node_list);
}

} // namespace svo

} // namespace octree_slam
