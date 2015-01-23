
#include <octree_slam/main.h>

std::string path_prefix = "../../";

int main(int argc, char** argv){

#if defined(__GNUC__)
  path_prefix = "";
#endif

	// Launch CUDA/GL
	if (init(argc, argv)) {

    frame_ = 0;
    seconds_ = time(NULL);
    fpstracker_ = 0;

		// GLFW main loop
		mainLoop();
	}

	system("PAUSE");
	return 0;
}


void mainLoop() {
	while(!glfwWindowShouldClose(window_)){
    //Read a frame from OpenNI Device
    if (DRAW_CAMERA_COLOR || DRAW_POINT_CLOUD) {
      camera_device_->readFrame();
    }

    camera_->update();

		if (USE_CUDA_RASTERIZER) {
      cuda_renderer_->rasterize(scene_->meshes()[0], scene_->textures()[0], camera_->camera(), lightpos_);
		} else if (DRAW_CAMERA_COLOR) {
      //Draw the current camera color frame to the window
      cuda_renderer_->pixelPassthrough(camera_device_->frame().color);
    } else if (DRAW_POINT_CLOUD) {
      gl_renderer_->renderPoints(camera_device_->frame(), camera_->camera());
    } else {
      gl_renderer_->rasterize(scene_->meshes()[0], camera_->camera(), lightpos_);
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

bool init(int argc, char* argv[]) {
  int choice = 2;
  std::cout << "Please enter which scene to load? '1'(dragon), '2'(cow), '3'(bunny)." << std::endl;
  std::cin >> choice;

  std::string local_path = path_prefix + "../objs/";
  std::string data = local_path;
  if (choice == 1)
    data += "dragon_tex.obj";
  else if (choice == 2)
    data += "cow_tex.obj";
  else if (choice == 3)
    data += "bunny_tex.obj";

  //Create scene
  scene_ = new octree_slam::world::Scene(path_prefix);

  //Load obj file
  scene_->loadObjFile(data);

  //Read texture
  scene_->loadBMP(path_prefix + std::string("../textures/texture1.bmp"));

  //Voxelize the scene
  if (VOXELIZE) {
    scene_->voxelizeMeshes(OCTREE);
  }

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
  if (DRAW_CAMERA_COLOR || DRAW_POINT_CLOUD) {
    camera_device_ = new octree_slam::sensor::OpenNIDevice();
  }

	// Initialize renderers
	if (USE_CUDA_RASTERIZER || DRAW_CAMERA_COLOR) {
    int width = DRAW_CAMERA_COLOR ? camera_device_->frameWidth() : width_;
    int height = DRAW_CAMERA_COLOR ? camera_device_->frameHeight() : height_;
    cuda_renderer_ = new octree_slam::rendering::CUDARenderer(VOXELIZE, path_prefix, width, height);
	} else if (DRAW_POINT_CLOUD || (!USE_CUDA_RASTERIZER && !DRAW_CAMERA_COLOR)) {
    gl_renderer_ = new octree_slam::rendering::OpenGLRenderer(VOXELIZE, path_prefix);
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
