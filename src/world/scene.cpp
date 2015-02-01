
#include <algorithm>

// Octree-SLAM Dependency
#include <octree_slam/world/scene.h>
#include <octree_slam/world/voxelization/voxelization.h>
#include <octree_slam/world/octree.h>
#include <octree_slam/world/svo/svo.h>

namespace octree_slam {

namespace world {

Scene::Scene(const std::string& path_prefix) {
  obj* o = new obj();
  objLoader loader(path_prefix + "../objs/cube.obj", o);
  o->buildVBOs();
  cube_ = objToMesh(o);
  delete o;
  for (int i = 0; i < 6; i++) {
    bbox_[i] = 0.0f;
  }
}

Scene::~Scene() {

}

void Scene::loadObjFile(const std::string& filename) {
  obj* o = new obj();
  objLoader loader(filename, o);
  o->buildVBOs();
  updateBBox(o->getBoundingBox()[0], o->getBoundingBox()[1], o->getBoundingBox()[18], 
    o->getBoundingBox()[8], o->getBoundingBox()[5], o->getBoundingBox()[2]);
  Mesh mesh = objToMesh(o);
  meshes_.push_back(mesh);
  delete o;
}

void Scene::loadBMP(const std::string& filename) {
  bmp_texture tex;
  int i;
  FILE* f = fopen(filename.c_str(), "rb");
  unsigned char info[54];
  fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

  // extract image height and width from header
  int width = *(int*)&info[18];
  int height = *(int*)&info[22];

  int size = 3 * width * height;
  unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
  fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
  fclose(f);
  glm::vec3 *color_data = new glm::vec3[size / 3];
  for (i = 0; i < size; i += 3){
    color_data[i / 3].r = (int)data[i + 2] / 255.0f;
    color_data[i / 3].g = (int)data[i + 1] / 255.0f;
    color_data[i / 3].b = (int)data[i] / 255.0f;
  }
  delete[]data;
  tex.data = color_data;
  tex.height = height;
  tex.width = width;

  textures_.push_back(tex);
}

void Scene::voxelizeMeshes(const bool octree) {
  //TODO: Voxelize all meshes
  if (meshes_.size() == 0) {
    return;
  }

  //For now, just voxelize the first mesh
  voxelization::setWorldSize(bbox_[0], bbox_[1], bbox_[2], bbox_[3], bbox_[4], bbox_[5]);

  Mesh m;
  if (!octree) {
    //voxelization::voxelizeToCubes(meshes_[0], &textures_[0], cube_, m);
    voxelization::voxelizeToGrid(meshes_[0], &textures_[0], voxel_grid_);
  } else {
    svo::voxelizeSVOCubes(meshes_[0], &textures_[0], cube_, m);
    meshes_[0] = m;
  }
}

Mesh Scene::objToMesh(obj* object) {

  Mesh m;
  m.vbo = object->getVBO();
  m.vbosize = object->getVBOsize();
  m.nbo = object->getNBO();
  m.nbosize = object->getNBOsize();
  m.cbo = object->getCBO();
  m.cbosize = object->getCBOsize();
  m.ibo = object->getIBO();
  m.ibosize = object->getIBOsize();
  m.tbo = object->getTBO();
  m.tbosize = object->getTBOsize();

  return m;
}

void Scene::updateBBox(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) {
  bbox_[0] = std::min(x1, bbox_[0]);
  bbox_[1] = std::min(y1, bbox_[1]);
  bbox_[2] = std::min(z1, bbox_[2]);
  bbox_[3] = std::max(x2, bbox_[3]);
  bbox_[4] = std::max(y2, bbox_[4]);
  bbox_[5] = std::max(z2, bbox_[5]);
}

} // namespace world

} // namespace octree_slam
