
#include <octree_slam/main.h>

string path_prefix = "../../";

int main(int argc, char** argv){

#if defined(__GNUC__)
  path_prefix = "";
#endif

	int choice = 2;
	cout << "Please enter which scene to load? '1'(dragon), '2'(cow), '3'(bunny)." << endl;
	cin >> choice;

	string local_path = path_prefix + "../objs/";
	string data = local_path;
	if (choice == 1)
		data += "dragon_tex.obj";
	else if (choice == 2)
		data += "cow_tex.obj";
  else if (choice == 3)
    data += "bunny_tex.obj";

	mesh_ = new obj();
	objLoader* loader = new objLoader(data, mesh_);
	mesh_->buildVBOs();
	meshes_.push_back(mesh_);
	delete loader;

  //Read texture
  readBMP((path_prefix + string("../textures/texture1.bmp")).c_str(), tex_);

	frame_ = 0;
	seconds_ = time (NULL);
	fpstracker_ = 0;

	// Launch CUDA/GL
	if (init(argc, argv)) {

		//Voxelize the scene
    mesh_geom_ = buildScene();

		// GLFW main loop
		mainLoop();
	}

	system("PAUSE");
	return 0;
}


void mainLoop() {
	while(!glfwWindowShouldClose(window_)){
    //Read a frame from OpenNI Device
    if (DRAW_CAMERA_COLOR) {
      camera_device_->readFrame();
    }

    camera_->update();

		if (USE_CUDA_RASTERIZER && !DRAW_CAMERA_COLOR) {
      cuda_renderer_->render(mesh_geom_, tex_, camera_->camera(), lightpos_);
		} else if (!DRAW_CAMERA_COLOR) {
      gl_renderer_->render(mesh_geom_, camera_->camera(), lightpos_);
    } else {
      //Draw the current camera color frame to the window
      camera_device_->drawColor();
      cuda_renderer_->render(mesh_geom_, tex_, camera_->camera(), lightpos_);
    }
    frame_++;
    fpstracker_++;

		time_t seconds2 = time (NULL);

		if(seconds2-seconds_ >= 1){

			fps_ = fpstracker_/(seconds2-seconds_);
			fpstracker_ = 0;
			seconds_ = seconds2;
		}

		string title = "Octree-SLAM | " + utilityCore::convertIntToString((int)fps_) + " FPS";
		glfwSetWindowTitle(window_, title.c_str());

		glfwSwapBuffers(window_);
	}
	glfwDestroyWindow(window_);
	glfwTerminate();
}

Mesh buildScene() {

	//Construct target mesh
	Mesh m_in;
	m_in.vbo = mesh_->getVBO();
	m_in.vbosize = mesh_->getVBOsize();
	m_in.nbo = mesh_->getNBO();
	m_in.nbosize = mesh_->getNBOsize();
	m_in.cbo = mesh_->getCBO();
	m_in.cbosize = mesh_->getCBOsize();
	m_in.ibo = mesh_->getIBO();
	m_in.ibosize = mesh_->getIBOsize();
  m_in.tbo = mesh_->getTBO();
  m_in.tbosize = mesh_->getTBOsize();

  if (!VOXELIZE) {
    return m_in;
  }

	//Load cube
	Mesh m_cube;
	obj* cube = new obj();
	string cubeFile = path_prefix + "../objs/cube.obj";
	objLoader* loader = new objLoader(cubeFile, cube);
	cube->buildVBOs();
	m_cube.vbo = cube->getVBO();
	m_cube.vbosize = cube->getVBOsize();
	m_cube.nbo = cube->getNBO();
	m_cube.nbosize = cube->getNBOsize();
	m_cube.cbo = cube->getCBO();
	m_cube.cbosize = cube->getCBOsize();
	m_cube.ibo = cube->getIBO();
	m_cube.ibosize = cube->getIBOsize();
	delete cube;

	//Voxelize
  Mesh mesh_geom;
  if (OCTREE){
    octree_slam::svo::voxelizeSVOCubes(m_in, &tex_, m_cube, mesh_geom);
  } else {
    octree_slam::voxelization::voxelizeToCubes(m_in, &tex_, m_cube, mesh_geom);
  }

  return mesh_geom;
}

//read .bmp texture and assign it to tex
void readBMP(const char* filename, bmp_texture &tex)
{
	int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);
	  glm::vec3 *color_data = new glm::vec3[size/3];
    for(i = 0; i < size; i += 3){
			color_data[i/3].r = (int)data[i+2]/255.0f;
			color_data[i/3].g = (int)data[i+1]/255.0f;
			color_data[i/3].b = (int)data[i]/255.0f;
    }
    delete []data;
	tex.data = color_data;
	tex.height = height;
	tex.width = width;
}


bool init(int argc, char* argv[]) {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		return false;
	}

	window_ = glfwCreateWindow(width_, height_, "Octree-SLAM", NULL, NULL);
	if (!window_){
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window_);

  //Create the virtual camera controller
  camera_ = new octree_slam::rendering::GLFWCameraController(window_, width_, height_);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if(glewInit()!=GLEW_OK){
		return false;
	}

  //Initialize camera rendering
  if (DRAW_CAMERA_COLOR) {
    camera_device_ = new octree_slam::sensor::OpenNIDevice();
    camera_device_->initPBO();
  }

	// Initialize renderers
	if (USE_CUDA_RASTERIZER) {
    int width = DRAW_CAMERA_COLOR ? camera_device_->colorFrameWidth() : width_;
    int height = DRAW_CAMERA_COLOR ? camera_device_->colorFrameHeight() : height_;
    cuda_renderer_ = new octree_slam::rendering::CUDARenderer(VOXELIZE, path_prefix, width, height);
    if (DRAW_CAMERA_COLOR) cuda_renderer_->setPBO(camera_device_->pbo());
	} else {
    gl_renderer_ = new octree_slam::rendering::OpenGLRenderer(VOXELIZE, path_prefix);
	}

  // Initialize Voxelization
  if (VOXELIZE) {
    octree_slam::voxelization::setWorldSize(mesh_->getBoundingBox()[0], mesh_->getBoundingBox()[1], mesh_->getBoundingBox()[18], 
                                            mesh_->getBoundingBox()[8], mesh_->getBoundingBox()[5], mesh_->getBoundingBox()[2]);
  }

	return true;
}

void shut_down(int return_code){
	cudaDeviceReset();
#ifdef __APPLE__
	glfwTerminate();
#endif
	exit(return_code);
}

void errorCallback(int error, const char* description){
	fputs(description, stderr);
}
