
// GL Dependencies
#include <glm/glm.hpp>
#include <GL/glut.h>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// Voxelpipe Dependency
#include <voxelpipe/voxelpipe.h>

// Octree-SLAM Dependencies
#include <octree_slam/timing_utils.h>
#include <octree_slam/world/voxelization/voxelization.h>
#include <octree_slam/cuda_common_kernels.h>

namespace octree_slam {

namespace voxelization {

//Declare voxelization resolution
#define GRID_RES 8
#define TILE_SIZE 3

__host__ __device__ int log_N() {
  return GRID_RES;
}
__host__ __device__ int log_T() {
  return TILE_SIZE;
}
//N is the total number of voxels (per dimension)
__host__ __device__ int N() {
  return 1 << log_N();
}
//M is the total number of tiles (per dimension)
__host__ __device__ int M() {
  return 1 << (log_N() - log_T());
}
//T is the tile size - voxels per tile (per dimension)
__host__ __device__ int T() {
  return 1 << log_T();
}

voxelpipe::FRContext<GRID_RES, TILE_SIZE>*  context;
bool first_time = true;

//Utility function for computing voxel sizes
__device__ void getSizes(const glm::vec3& bbox0, const glm::vec3& bbox1, glm::vec3& t_d, glm::vec3& p_d) {
  //Compute tile/grid sizes
  t_d = glm::vec3((bbox1.x - bbox0.x) / float(M()),
    (bbox1.y - bbox0.y) / float(M()),
    (bbox1.z - bbox0.z) / float(M()));
  p_d = glm::vec3(t_d.x / float(T()),
    t_d.y / float(T()), t_d.z / float(T()));
}

__device__ glm::vec3 getCenterFromIndex(int idx, const glm::vec3& bbox0, const glm::vec3& bbox1) {
  glm::vec3 p_d, t_d;
  getSizes(bbox0, bbox1, t_d, p_d);
  int T3 = T()*T()*T();
  int tile_num = idx / T3;
  int pix_num = idx % T3;
  glm::vec3 cent;
  int tz = tile_num / (M()*M()) % M();
  int pz = pix_num / (T()*T()) % T();
  int ty = tile_num / M() % M();
  int py = pix_num / T() % T();
  int tx = tile_num % M();
  int px = pix_num % T();
  cent.x = bbox0.x + tx*t_d.x + px*p_d.x + p_d.x / 2.0f;
  cent.y = bbox0.y + ty*t_d.y + py*p_d.y + p_d.y / 2.0f;
  cent.z = bbox0.z + tz*t_d.z + pz*p_d.z + p_d.z / 2.0f;
  return cent;
}

__host__ float computeScale(const glm::vec3& bbox0, const glm::vec3& bbox1) {
  return (bbox1.x - bbox0.x)/float(N())/2.0f;
}

struct ColorShader
{
  glm::vec3* texture;
  int tex_width;
  int tex_height;
  float* texcoord;
  int texcoord_size;

  __device__ float shade(
    const int tri_id,
    const float4 v0,
    const float4 v1,
    const float4 v2,
    const float3 n,
    const float  bary0,
    const float  bary1,
    const int3   xyz) const
  {
    //If there is no texture, just return green
    if (tex_width == 0) {
      return __int_as_float((255 << 8) + (127 << 24));
    }

    //If there are no texcoordinates, just return the first value in the texture
    if (texcoord_size == 0) {
      int r = (int)(texture[0].r * 255.0);
      int g = (int)(texture[0].g * 255.0);
      int b = (int)(texture[0].b * 255.0);
      return __int_as_float(r+(g << 8) + (b << 16) + (127 << 24));
    }

    //Get the texture coordinates from the triangle id
    int t1_x = texcoord[6 * tri_id] * tex_width;
    int t1_y = texcoord[6 * tri_id + 1] * tex_height;
    int t2_x = texcoord[6 * tri_id + 2] * tex_width;
    int t2_y = texcoord[6 * tri_id + 3] * tex_height;
    int t3_x = texcoord[6 * tri_id + 4] * tex_width;
    int t3_y = texcoord[6 * tri_id + 5] * tex_height;

    //Get the colors from the texture at these vertices
    glm::vec3 c1 = texture[t1_y * tex_width + t1_x];
    glm::vec3 c2 = texture[t2_y * tex_width + t2_x];
    glm::vec3 c3 = texture[t3_y * tex_width + t3_x];

    //TODO: Interpolate using barycentric coordinates
    glm::vec3 color = c1;

    //Compute rgb components
    int r = (int) (clamp(color.r, 0.0f, 1.0f) * 255.0f);
    int g = (int) (clamp(color.g, 0.0f, 1.0f) * 255.0f);
    int b = (int) (clamp(color.b, 0.0f, 1.0f) * 255.0f);

    //Compact
    int val = r + (g << 8) + (b << 16) + (127 << 24);

    return __int_as_float(val);
  }
};

__global__ void getOccupiedVoxels(void* fb, int M, int T, int* voxels) {
  int T3 = T*T*T;
  int M3 = M*M*M;

  int pix_num = (blockIdx.x * 256 % T3) + threadIdx.x;
  int tile_num = blockIdx.x * 256 / T3;

  if (pix_num < T3 && tile_num < M3) {
    //TODO: Is there any benefit in making this shared?
    float* tile;

    bool is_occupied;
    tile = (float*)fb + tile_num*T3;
    int alpha = __float_as_int(tile[pix_num]) >> 24;
    is_occupied = alpha > 0;

    if (is_occupied) {
      voxels[tile_num*T3 + pix_num] = tile_num*T3 + pix_num;
    } else {
      voxels[tile_num*T3 + pix_num] = -1;
    }
  }

}

//Thrust predicate for removal of empty voxels
struct check_voxel {
  __host__ __device__
    bool operator() (const int& c) {
    return (c != -1);
  }
};

__global__ void extractValues(void* fb, int* voxels, int num_voxels, int* values) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_voxels) {
    //TODO: Make this support other storage_type's besides int32
    float* tile = (float*)fb;
    values[index] = __float_as_int(tile[voxels[index]]);
  }
}

__global__ void createCubeMesh(const glm::vec4* voxels, const glm::vec4* values, const float scale_factor, const int num_voxels, float* cube_vbo, 
                                int cube_vbosize, int* cube_ibo, int cube_ibosize, float* cube_nbo, float* out_vbo, int* out_ibo, float* out_nbo, float* out_cbo) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < num_voxels) {

    int vbo_offset = idx * cube_vbosize;
    int ibo_offset = idx * cube_ibosize;
    glm::vec4 center = voxels[idx];
    glm::vec4 color = values[idx];

    for (int i = 0; i < cube_vbosize; i++) {
      if (i % 3 == 0) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.x;
        out_cbo[vbo_offset + i] = color.r;
      } else if (i % 3 == 1) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.y;
        out_cbo[vbo_offset + i] = color.g;
      } else {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.z;
        out_cbo[vbo_offset + i] = color.b;
      }
      out_nbo[vbo_offset + i] = cube_nbo[i];
    }

    for (int i = 0; i < cube_ibosize; i++) {
      out_ibo[ibo_offset + i] = cube_ibo[i] + ibo_offset;
    }

  }

}

__global__ void createVoxelGrid(const int* voxels, const int* values, const glm::vec3 bbox0, const glm::vec3 bbox1, const int num_voxels, glm::vec4* centers, glm::vec4* colors) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < num_voxels) {

    glm::vec3 center = getCenterFromIndex(voxels[idx], bbox0, bbox1);
    centers[idx] = glm::vec4(center.x, center.y, center.z, 1.0f);

    int color = values[idx];
    colors[idx].r = (float)((color & 0xFF) / 255.0);
    colors[idx].g = (float)(((color >> 8) & 0xFF) / 255.0);
    colors[idx].b = (float)(((color >> 16) & 0xFF) / 255.0);

  }

}

__host__ int voxelizeMesh(const Mesh &m_in, const bmp_texture* h_tex, const BoundingBox& box, int* d_voxels, int* d_values) {

  //Initialize sizes
  const int n_triangles = m_in.ibosize / 3;
  const int n_vertices = m_in.vbosize / 3;

  //Create host vectors
  thrust::host_vector<int4> h_triangles(n_triangles);
  thrust::host_vector<float4> h_vertices(n_vertices);

  //Fill in the data
  for (int i = 0; i < n_vertices; i++) {
    h_vertices[i].x = m_in.vbo[i * 3 + 0];
    h_vertices[i].y = m_in.vbo[i * 3 + 1];
    h_vertices[i].z = m_in.vbo[i * 3 + 2];
  }
  for (int i = 0; i < n_triangles; i++) {
    h_triangles[i].x = m_in.ibo[i * 3 + 0];
    h_triangles[i].y = m_in.ibo[i * 3 + 1];
    h_triangles[i].z = m_in.ibo[i * 3 + 2];
  }

  //Copy to device vectors
  thrust::device_vector<int4> d_triangles(h_triangles);
  thrust::device_vector<float4> d_vertices(h_vertices);

  if (first_time) {
    //Create the voxelpipe context
    context = new voxelpipe::FRContext<GRID_RES, TILE_SIZE>();

    //Reserve data for voxelpipe
    context->reserve(n_triangles, 1024u * 1024u * 16u);
  }
  first_time = false;

  //Initialize the result data on the device
  thrust::device_vector<float>  d_fb(M()*M()*M() * T()*T()*T());

  //Copy the texture to the device
  glm::vec3 *device_tex = NULL;
  cudaMalloc((void**)&device_tex, h_tex->width * h_tex->height *sizeof(glm::vec3));
  cudaMemcpy(device_tex, h_tex->data, h_tex->width * h_tex->height *sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //Copy the texture coordinates to the device
  float* device_texcoord = NULL;
  cudaMalloc((void**)&device_texcoord, m_in.tbosize * sizeof(float));
  cudaMemcpy(device_texcoord, m_in.tbo, m_in.tbosize *sizeof(float), cudaMemcpyHostToDevice);

  //Create the shader to be used that will write texture colors to voxels
  ColorShader my_shader;
  my_shader.texture = device_tex;
  my_shader.tex_height = h_tex->height;
  my_shader.tex_width = h_tex->width;
  my_shader.texcoord = device_texcoord;
  my_shader.texcoord_size = m_in.tbosize;

  //Perform coarse and fine voxelization
  context->coarse_raster(n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), 
    make_float3(box.bbox0.x, box.bbox0.y, box.bbox0.z), make_float3(box.bbox1.x, box.bbox1.y, box.bbox1.z));
  context->fine_raster< voxelpipe::Float, voxelpipe::FP32S_FORMAT, voxelpipe::THIN_RASTER, voxelpipe::NO_BLENDING, ColorShader >(
    n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), 
    make_float3(box.bbox0.x, box.bbox0.y, box.bbox0.z), make_float3(box.bbox1.x, box.bbox1.y, box.bbox1.z), thrust::raw_pointer_cast(&d_fb.front()), my_shader);

  cudaFree(device_tex);
  cudaFree(device_texcoord);

  //Get voxel centers
  int numVoxels = N()*N()*N();
  int* d_vox;
  cudaMalloc((void**)&d_vox, numVoxels*sizeof(int));
  getOccupiedVoxels<<<N()*N()*N(), 256>>>(thrust::raw_pointer_cast(&d_fb.front()), M(), T(), d_vox);
  cudaDeviceSynchronize();

  //Stream Compact voxels to remove the empties
  numVoxels = thrust::copy_if(thrust::device_pointer_cast(d_vox), thrust::device_pointer_cast(d_vox) + numVoxels, thrust::device_pointer_cast(d_voxels), check_voxel()) - thrust::device_pointer_cast(d_voxels);

  std::cout << "Num Voxels: " << numVoxels << std::endl;

  //Extract the values at these indices
  extractValues<<<ceil((float)numVoxels / 256.0f), 256 >>>(thrust::raw_pointer_cast(&d_fb.front()), d_voxels, numVoxels, d_values);
  cudaDeviceSynchronize();

  cudaFree(d_vox);

  return numVoxels;
}

__host__ void voxelGridToMesh(const VoxelGrid& grid, const Mesh &m_cube, Mesh &m_out) {

  //Move cube data to GPU
  thrust::device_vector<float> d_vbo_cube(m_cube.vbo, m_cube.vbo + m_cube.vbosize);
  thrust::device_vector<int> d_ibo_cube(m_cube.ibo, m_cube.ibo + m_cube.ibosize);
  thrust::device_vector<float> d_nbo_cube(m_cube.nbo, m_cube.nbo + m_cube.nbosize);

  //Create output structs
  float* d_vbo_out;
  int* d_ibo_out;
  float* d_nbo_out;
  float* d_cbo_out;
  cudaMalloc((void**)&d_vbo_out, grid.size * m_cube.vbosize * sizeof(float));
  cudaMalloc((void**)&d_ibo_out, grid.size * m_cube.ibosize * sizeof(int));
  cudaMalloc((void**)&d_nbo_out, grid.size * m_cube.nbosize * sizeof(float));
  cudaMalloc((void**)&d_cbo_out, grid.size * m_cube.nbosize * sizeof(float));

  //Warn if vbo and nbo are not same size on cube
  if (m_cube.vbosize != m_cube.nbosize) {
    std::cout << "ERROR: cube vbo and nbo have different sizes." << std::endl;
    return;
  }

  //Create resulting cube-ized mesh
  createCubeMesh<<<ceil((float)grid.size / 256.0f), 256>>>(grid.centers, grid.colors, computeScale(grid.bbox.bbox0, grid.bbox.bbox1) / CUBE_MESH_SCALE, grid.size, thrust::raw_pointer_cast(&d_vbo_cube.front()),
    m_cube.vbosize, thrust::raw_pointer_cast(&d_ibo_cube.front()), m_cube.ibosize, thrust::raw_pointer_cast(&d_nbo_cube.front()), d_vbo_out, d_ibo_out, d_nbo_out, d_cbo_out);

  //Store output sizes
  m_out.vbosize = grid.size * m_cube.vbosize;
  m_out.ibosize = grid.size * m_cube.ibosize;
  m_out.nbosize = grid.size * m_cube.nbosize;
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
  cudaFree(d_cbo_out);
}

__host__ void meshToVoxelGrid(const Mesh &m_in, const bmp_texture* tex, VoxelGrid &grid_out) {
  //Voxelize the mesh input
  int numVoxels = N()*N()*N();
  int* d_voxels;
  int* d_values;
  cudaMalloc((void**)&d_voxels, numVoxels*sizeof(int));
  cudaMalloc((void**)&d_values, numVoxels*sizeof(int));
  numVoxels = voxelizeMesh(m_in, tex, m_in.bbox, d_voxels, d_values);

  //Extract centers and colors
  cudaMalloc((void**)&(grid_out.centers), numVoxels*sizeof(glm::vec4));
  cudaMalloc((void**)&(grid_out.colors), numVoxels*sizeof(glm::vec4));
  createVoxelGrid<<<ceil((float)numVoxels / 256.0f), 256>>>(d_voxels, d_values, m_in.bbox.bbox0, m_in.bbox.bbox1, numVoxels, grid_out.centers, grid_out.colors);

  //Free old memory from the grid
  cudaFree(d_voxels);
  cudaFree(d_values);

  //Set the scale and size
  grid_out.scale = computeScale(m_in.bbox.bbox0, m_in.bbox.bbox1);
  grid_out.size = numVoxels;

  //Copy the bounding box
  grid_out.bbox = m_in.bbox;
}

} // namespace voxelization

} // namespace octree_slam

