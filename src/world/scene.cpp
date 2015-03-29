
#include <algorithm>

// Octree-SLAM Dependency
#include <octree_slam/world/scene.h>
#include <octree_slam/world/voxelization/voxelization.h>
#include <octree_slam/world/octree.h>

namespace octree_slam {

namespace world {

Scene::Scene() : 
tree_(NULL),
voxel_grid_(new VoxelGrid()) {
  for (int i = 0; i < 6; i++) {
    bbox_[i] = 0.0f;
  }
}

Scene::~Scene() {
  if (tree_) {
    delete tree_;
  }
}

void Scene::loadObjFile(const std::string& filename) {
  obj* o = new obj();
  objLoader loader(filename, o);
  o->buildVBOs();
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
  //TODO: Voxelize all meshes, not just the first one
  if (meshes_.size() == 0) {
    return;
  }

  if (!octree) {
    voxelization::meshToVoxelGrid(meshes_[0], &textures_[0], *voxel_grid_);
    voxel_grid_->scale = meshes_[0].bbox.bbox1.x / (1 << voxelization::log_N());
  } else {
    VoxelGrid grid;
    voxelization::meshToVoxelGrid(meshes_[0], &textures_[0], grid);
    voxel_grid_->scale = meshes_[0].bbox.bbox1.x / (1 << voxelization::log_N());
    if (!tree_) {
      tree_ = new Octree(voxel_grid_->scale, (meshes_[0].bbox.bbox1 + meshes_[0].bbox.bbox0) / 2.0f, meshes_[0].bbox.bbox1.x);
    }
    tree_->addVoxelGrid(grid);
    voxel_grid_->bbox = tree_->boundingBox();
    voxel_grid_->scale *= (float) 1; //Use this to render higher levels in the octree
    tree_->extractVoxelGrid(*voxel_grid_);
  }
}

void Scene::extractVoxelGridFromOctree() {
  //Clean up the current voxel grid
  delete voxel_grid_;
  voxel_grid_ = new VoxelGrid();

  //TODO: Take in a "resolution" parameter for the extraction process
  voxel_grid_->bbox = tree_->boundingBox();
  voxel_grid_->scale = 0.01f;
  tree_->extractVoxelGrid(*voxel_grid_);
}

void Scene::addPointCloudToOctree(const glm::vec3& origin, const glm::vec3* points, const Color256* colors, const int size, const BoundingBox& bbox) {
  //Create the octree if needed
  if (!tree_) {
    tree_ = new Octree(0.01f, (bbox.bbox1 + bbox.bbox0) / 2.0f, bbox.bbox1.x);

  //Expand the octree if needed
  } else if (!tree_->boundingBox().contains(bbox)) {
    //Determine the necessary expansion size of the octree and expand it
    float new_size = bbox.distanceOutside(tree_->boundingBox());
    tree_->expandBySize(new_size);
  }

  //Add the points to the octree
  tree_->addCloud(origin, points, colors, size, bbox);

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

  m.bbox.bbox0 = glm::vec3(object->getBoundingBox()[0], object->getBoundingBox()[1], object->getBoundingBox()[18]);
  m.bbox.bbox1 = glm::vec3(object->getBoundingBox()[8], object->getBoundingBox()[5], object->getBoundingBox()[2]);

  return m;
}

} // namespace world

} // namespace octree_slam
