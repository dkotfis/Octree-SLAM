
#include <stack>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

// Octree-SLAM Dependencies
#include <octree_slam/timing_utils.h>
#include <octree_slam/world/voxelization/voxelization_utils.h>
#include <octree_slam/world/svo/svo.h>

namespace octree_slam {

namespace svo {

texture<float, 3> brick_tex;
surface<void, 3> brick_surf;

__global__ void flagNodes(int* voxels, int numVoxels, int* octree, int M, int T, float3 bbox0, float3 t_d, float3 p_d, int tree_depth) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index < numVoxels) {
    float3 center = voxelization::getCenterFromIndex(voxels[index], M, T, bbox0, t_d, p_d);
    float edge_length = abs(bbox0.x);
    float3 center_depth = make_float3(0.0f, 0.0f, 0.0f);
    int node_idx = 0;
    int this_node;

    //Loop until the specified depth
    for (int i = 0; i < tree_depth; i++) {
      int x = center.x > center_depth.x;
      int y = center.y > center_depth.y;
      int z = center.z > center_depth.z;
      this_node = node_idx + (x + (y << 1) + (z << 2));

      if (i < tree_depth - 1) {
        //The lowest 30 bits are the address
        node_idx = octree[2 * this_node] & 0x3FFFFFFF;

        //Update the center depth for the next iteration
        center_depth.x += edge_length / 2 * (x ? 1 : -1);
        center_depth.y += edge_length / 2 * (y ? 1 : -1);
        center_depth.z += edge_length / 2 * (z ? 1 : -1);
      }
      edge_length /= 2.0f;
    }
    octree[2 * this_node] = octree[2 * this_node] | 0x40000000;
  }

}

__global__ void splitNodes(int* octree, int* numNodes, int poolSize, int startNode) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index < poolSize) {
    int node = octree[2 * (index+startNode)];

    //Split the node if its flagged
    if (node & 0x40000000) {
      //Get a new node tile
      int newNode = atomicAdd(numNodes, 8);

      //Point this node at the new tile
      octree[2 * (index+startNode)] = (octree[2 * (index+startNode)] & 0xC0000000) | (newNode & 0x3FFFFFFF);

      //Initialize new child nodes to 0's
      for (int off = 0; off < 8; off++) {
        octree[2 * (newNode + off)] = 0;
        octree[2 * (newNode + off) + 1] = 0;
      }
    }
  }

}

__global__ void fillNodes(int* voxels, int numVoxels, int* values, int* octree, int M, int T, float3 bbox0, float3 t_d, float3 p_d) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index < numVoxels) {
    float3 center = voxelization::getCenterFromIndex(voxels[index], M, T, bbox0, t_d, p_d);
    float edge_length = abs(bbox0.x);
    float3 center_depth = make_float3(0.0f, 0.0f, 0.0f);
    int node_idx = 0;
    int this_node;
    bool has_child = true;

    //Loop until the specified depth
    while (has_child) {
      int x = center.x > center_depth.x;
      int y = center.y > center_depth.y;
      int z = center.z > center_depth.z;
      this_node = node_idx + (x + (y << 1) + (z << 2));

      has_child = octree[2 * this_node] & 0x40000000;

      if (has_child) {
        //The lowest 30 bits are the address
        node_idx = octree[2 * this_node] & 0x3FFFFFFF;

        //Update the center depth for the next iteration
        center_depth.x += edge_length / 2 * (x ? 1 : -1);
        center_depth.y += edge_length / 2 * (y ? 1 : -1);
        center_depth.z += edge_length / 2 * (z ? 1 : -1);
      }
      edge_length /= 2.0f;
    }
    octree[2 * this_node + 1] = values[index];
  }

}

__global__ void mipmapNodes(int* octree, int poolSize, int startNode) {

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

  //Get the child pointer
  int childPoint = (node & 0x3FFFFFFF);

  //Loop through children values and average them
   glm::vec4 val = glm::vec4(0.0);
  for (int i = 0; i < 8; i++) {
    int child_val = octree[2*(childPoint+i) + 1];
    val.r += (float) (child_val & 0xFF) / 8.0f;
    val.g += (float) ((child_val >> 8) & 0xFF) / 8.0f;
    val.b += (float) ((child_val >> 16) & 0xFF) / 8.0f;
    val.a += (float) ((child_val >> 24) & 0x7F) / 8.0f;
  }

  //Assign value of this node to the average
  int r = (int) (val.r);
  int g = (int) (val.g);
  int b = (int) (val.b);
  int a = (int) (val.a);
  octree[(2 * (index + startNode)) + 1] = r + (g << 8) + (b << 16) + (a << 24);

}

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

__global__ void createCubeMeshFromSVO(int* octree, int* counter, int total_depth, float3 bbox0, float cube_scale, int num_voxels, float* cube_vbo,
  int cube_vbosize, int* cube_ibo, int cube_ibosize, float* cube_nbo, float* out_vbo, int* out_ibo, float* out_nbo, float* out_cbo) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  float edge_length = abs(bbox0.x);
  float3 center = make_float3(0.0f, 0.0f, 0.0f);
  int pointer = 0;
  bool has_child = true;
  int val2;
  int depth = 0;

  while (has_child && depth++ < total_depth) {
    //Get the lowest 3 bits to encode the first move
    int pos = idx & 0x7;

    //Get the value from the octree
    int val = octree[2*(pointer+pos)];
    val2 = octree[2*(pointer + pos) + 1];

    //It it is not occupied, do not continue
    has_child = val & 0x40000000;

    //Don't continue if it does not have a child
    if (!has_child && (octree[2 * (pointer + pos) + 1] >> 24 == 0)) {
      return;
    }

    //Get the child pointer for the next depth
    pointer = val & 0x3FFFFFFF;

    //Decode the value into xyz
    int x = pos & 0x1;
    int y = pos & 0x2;
    int z = pos & 0x4;

    //Update the center
    center.x += edge_length / 2 * (x ? 1 : -1);
    center.y += edge_length / 2 * (y ? 1 : -1);
    center.z += edge_length / 2 * (z ? 1 : -1);

    //Half the edge length for the next iteration
    edge_length /= 2.0f;

    //Shift right for the next iteration
    idx = idx >> 3;
  }

  int vidx = atomicAdd(counter, 1);
  //TODO: Detect if we exceed the allocated memory, and break out + allocate more

  if (vidx < num_voxels) {

    int vbo_offset = vidx * cube_vbosize;
    int ibo_offset = vidx * cube_ibosize;

    for (int i = 0; i < cube_vbosize; i++) {
      if (i % 3 == 0) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * (edge_length/cube_scale) + center.x;
        out_cbo[vbo_offset + i] = (float)((val2 & 0xFF) / 255.0);
      }
      else if (i % 3 == 1) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * (edge_length/cube_scale) + center.y;
        out_cbo[vbo_offset + i] = (float)(((val2 >> 8) & 0xFF) / 255.0);
      }
      else {
        out_vbo[vbo_offset + i] = cube_vbo[i] *(edge_length/cube_scale) + center.z;
        out_cbo[vbo_offset + i] = (float)(((val2 >> 16) & 0xFF) / 255.0);
      }
      out_nbo[vbo_offset + i] = cube_nbo[i];
    }

    for (int i = 0; i < cube_ibosize; i++) {
      out_ibo[ibo_offset + i] = cube_ibo[i] + ibo_offset;
    }

  }

}

//This is based on Cyril Crassin's approach
extern "C" void svoFromVoxels(int* d_voxels, int numVoxels, int* d_values, int* d_octree, cudaArray* d_bricks) {
  int numNodes = 8;
  std::stack<int> startingNodes;
  startingNodes.push(0);
  int* d_numNodes;
  cudaMalloc((void**)&d_numNodes, sizeof(int));
  cudaMemcpy(d_numNodes, &numNodes, sizeof(int), cudaMemcpyHostToDevice);
  int depth = 0;

  //Get voxel grid params
  voxelization::gridParams params = voxelization::getParams();

  while (numNodes < (numVoxels*voxelization::log_N) && ++depth < voxelization::log_N) {

    //First, parallelize on voxels and flag nodes to be subdivided
    flagNodes<<<(numVoxels / 256) + 1, 256>>>(d_voxels, numVoxels, d_octree, voxelization::M, voxelization::T, params.bbox0, params.t_d, params.p_d, depth);

    cudaDeviceSynchronize();

    //Then, parallize on nodes and subdivide
    splitNodes<<<((numNodes - startingNodes.top()) / 256) + 1, 256>>>(d_octree, d_numNodes, numNodes - startingNodes.top(), startingNodes.top());
    startingNodes.push(numNodes);

    cudaDeviceSynchronize();
    cudaMemcpy(&numNodes, d_numNodes, sizeof(int), cudaMemcpyDeviceToHost);
  }

  std::cout << "Num Nodes: " << numNodes << std::endl;

  //Write voxel values into the lowest level of the svo
  fillNodes<<<(numVoxels / 256) + 1, 256>>>(d_voxels, numVoxels, d_values, d_octree, voxelization::M, voxelization::T, params.bbox0, params.t_d, params.p_d);
  cudaDeviceSynchronize();

  //Initialize the global brick counter
  float* d_numBricks;
  float numBricks = 0;
  if (USE_BRICK_POOL) {
    cudaMalloc((void**)&d_numBricks, sizeof(float));
    cudaMemcpy(d_numBricks, &numBricks, sizeof(float), cudaMemcpyHostToDevice);
  }

  //Loop through the levels of the svo bottom to top and map the values by averaging child values
  numNodes = startingNodes.top(); //Skip the lowest level since the fillNodes() kernel handled that level
  startingNodes.pop();
  while (!startingNodes.empty()) {
    if (USE_BRICK_POOL) {
      mipmapBricks << <((numNodes - startingNodes.top()) / 256) + 1, 256 >> >(d_octree, numNodes - startingNodes.top(), startingNodes.top(), d_bricks, d_numBricks);
    } else {
      mipmapNodes << <((numNodes - startingNodes.top()) / 256) + 1, 256 >> >(d_octree, numNodes - startingNodes.top(), startingNodes.top());
    }
    cudaDeviceSynchronize();

    numNodes = startingNodes.top();
    startingNodes.pop();
  }

  cudaFree(d_numNodes);
  if (USE_BRICK_POOL) {
    cudaFree(d_numBricks);
  }
}

extern "C" void extractCubesFromSVO(int* d_octree, int numVoxels, Mesh &m_cube, Mesh &m_out) {

  //Move cube data to GPU
  thrust::device_vector<float> d_vbo_cube(m_cube.vbo, m_cube.vbo + m_cube.vbosize);
  thrust::device_vector<int> d_ibo_cube(m_cube.ibo, m_cube.ibo + m_cube.ibosize);
  thrust::device_vector<float> d_nbo_cube(m_cube.nbo, m_cube.nbo + m_cube.nbosize);

  //Create output structs
  float* d_vbo_out;
  int* d_ibo_out;
  float* d_nbo_out;
  float* d_cbo_out;
  cudaMalloc((void**)&d_vbo_out, numVoxels * m_cube.vbosize * sizeof(float));
  cudaMalloc((void**)&d_ibo_out, numVoxels * m_cube.ibosize * sizeof(int));
  cudaMalloc((void**)&d_nbo_out, numVoxels * m_cube.nbosize * sizeof(float));
  cudaMalloc((void**)&d_cbo_out, numVoxels * m_cube.nbosize * sizeof(float));

  //Warn if vbo and nbo are not same size on cube
  if (m_cube.vbosize != m_cube.nbosize) {
    std::cout << "ERROR: cube vbo and nbo have different sizes." << std::endl;
    return;
  }

  //Create global counter to determine where to write the output
  int* d_counter;
  int initial_count = 0;
  cudaMalloc((void**)&d_counter, sizeof(int));
  cudaMemcpy(d_counter, &initial_count, sizeof(int), cudaMemcpyHostToDevice);

  //Determine how to scale the number of threads needed based on the octree depth to render
  int fac = (voxelization::log_N > log_SVO_N) ? pow(8, voxelization::log_N - log_SVO_N) : 1;

  //Get voxel grid params
  voxelization::gridParams params = voxelization::getParams();

  //Create resulting cube-ized mesh
  createCubeMeshFromSVO << <(voxelization::N*voxelization::N*voxelization::N / 256 / fac) + 1, 256 >> >(d_octree, d_counter, log_SVO_N, params.bbox0, voxelization::CUBE_MESH_SCALE, 
    numVoxels, thrust::raw_pointer_cast(&d_vbo_cube.front()), m_cube.vbosize, thrust::raw_pointer_cast(&d_ibo_cube.front()), m_cube.ibosize, thrust::raw_pointer_cast(&d_nbo_cube.front()), 
    d_vbo_out, d_ibo_out, d_nbo_out, d_cbo_out);

  //Store output sizes
  m_out.vbosize = numVoxels * m_cube.vbosize;
  m_out.ibosize = numVoxels * m_cube.ibosize;
  m_out.nbosize = numVoxels * m_cube.nbosize;
  m_out.cbosize = m_out.nbosize;

  //Memory allocation for the outputs
  m_out.vbo = (float*)malloc(m_out.vbosize * sizeof(float));
  m_out.ibo = (int*)malloc(m_out.ibosize * sizeof(int));
  m_out.nbo = (float*)malloc(m_out.nbosize * sizeof(float));
  m_out.cbo = (float*)malloc(m_out.cbosize * sizeof(float));

  //Sync here after doing some CPU work
  cudaDeviceSynchronize();

  //Copy data back from GPU
  //TODO: Can we avoid this step by making everything run from device-side VBO/IBO/NBO/CBO?
  cudaMemcpy(m_out.vbo, d_vbo_out, m_out.vbosize*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.ibo, d_ibo_out, m_out.ibosize*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.nbo, d_nbo_out, m_out.nbosize*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.cbo, d_cbo_out, m_out.cbosize*sizeof(float), cudaMemcpyDeviceToHost);

  ///Free GPU memory
  cudaFree(d_vbo_out);
  cudaFree(d_ibo_out);
  cudaFree(d_nbo_out);
  cudaFree(d_counter);
}

extern "C" void voxelizeSVOCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out) {

  //Voxelize the mesh input
  int numVoxels = voxelization::N * voxelization::N * voxelization::N;
  int* d_voxels;
  int* d_values;
  cudaMalloc((void**)&d_voxels, numVoxels*sizeof(int));
  cudaMalloc((void**)&d_values, numVoxels*sizeof(int));
  numVoxels = voxelization::voxelizeMesh(m_in, tex, d_voxels, d_values);

  //Create the octree (with the brick pool)
  int* d_octree = NULL;
  cudaMalloc((void**)&d_octree, 8 * voxelization::log_N * numVoxels * sizeof(int));
  cudaArray* d_bricks = NULL;
  cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
  cudaExtent extents;
  extents.width = 3*numVoxels;
  extents.height = 3;
  extents.depth = 3;
  if (USE_BRICK_POOL) {
    cudaMalloc3DArray(&d_bricks, &channel, extents, cudaArraySurfaceLoadStore);
    cudaBindTextureToArray(brick_tex, d_bricks);
    cudaBindSurfaceToArray(brick_surf, d_bricks);
  }
  svoFromVoxels(d_voxels, numVoxels, d_values, d_octree, d_bricks);

  //Extract cubes from the leaves of the octree
  extractCubesFromSVO(d_octree, numVoxels, m_cube, m_out);

  //Free up GPU memory
  cudaFree(d_voxels);
  cudaFree(d_octree);
  if (USE_BRICK_POOL) {
    cudaFreeArray(d_bricks);
  }

}

} // namespace svo

} // namespace octree_slam
