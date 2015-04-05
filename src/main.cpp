
#include <octree_slam/main.h>
#include <glm/gtc/matrix_transform.hpp>

#include <ctime>
#include <iostream>

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
    //Read a frame from OpenNI Device and predict the camera's motion
    camera_device_->readFrame();
    //camera_estimation_->update(camera_device_->rawFrame());

    //Transform the points and add to the octree
    int num_points = camera_device_->rawFrame()->height*camera_device_->rawFrame()->width;
    octree_slam::sensor::generateVertexMap(camera_device_->rawFrame()->depth, points_, camera_device_->frameWidth(), camera_device_->frameHeight(), camera_device_->focalLength(), make_int2(camera_device_->frameWidth(), camera_device_->frameHeight()));
    octree_slam::sensor::transformVertexMap(points_, glm::mat4(camera_estimation_->orientation()) * glm::translate(glm::mat4(1.0f), camera_estimation_->position()), camera_device_->frameWidth()*camera_device_->frameHeight());
    cudaDeviceSynchronize();
    BoundingBox cloud_bbox;
    octree_slam::sensor::computePointCloudBoundingBox(points_, num_points, cloud_bbox);
    scene_->addPointCloudToOctree(camera_estimation_->position(), points_, camera_device_->rawFrame()->color, num_points, cloud_bbox);

    //Update the virtual camera based on keyboard/mouse input
    camera_->update();

    //Render the appropriate data
		if (USE_CUDA_RASTERIZER) {
      cuda_renderer_->rasterize(scene_->meshes()[0], scene_->textures()[0], camera_->camera(), lightpos_);
		} else if (DRAW_CAMERA_COLOR) {
      cuda_renderer_->pixelPassthrough(camera_device_->rawFrame()->color);
    } else if (DRAW_POINT_CLOUD) {
      gl_renderer_->renderPoints(points_, camera_device_->rawFrame()->color, camera_device_->frameWidth()*camera_device_->frameHeight(), camera_->camera());
    } else if (CONE_TRACING) {
      BoundingBox render_volume = cloud_bbox; //TODO: compute this properly using the camera position
      cuda_renderer_->coneTraceSVO(scene_->svo(render_volume), camera_->camera(), lightpos_);
    } else if (OCTREE) {
      scene_->extractVoxelGridFromOctree();
      gl_renderer_->rasterizeVoxels(scene_->voxel_grid(), camera_->camera(), lightpos_);
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

  /*
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
  */
  //Create scene
  scene_ = new octree_slam::world::Scene();

  //Load obj file
  //scene_->loadObjFile(data);

  //Read texture
  //scene_->loadBMP(path_prefix + std::string("../textures/texture1.bmp"));
  

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
  camera_device_ = new octree_slam::sensor::OpenNIDevice();
  camera_estimation_ = new octree_slam::sensor::RGBDCamera(camera_device_->frameWidth(), camera_device_->frameHeight(), camera_device_->focalLength());
  cudaMalloc((void**)&points_, camera_device_->frameWidth()*camera_device_->frameHeight()*sizeof(glm::vec3));

	// Initialize renderers
	if (USE_CUDA_RASTERIZER || DRAW_CAMERA_COLOR || CONE_TRACING) {
    int width = DRAW_CAMERA_COLOR ? camera_device_->frameWidth() : width_;
    int height = DRAW_CAMERA_COLOR ? camera_device_->frameHeight() : height_;
    cuda_renderer_ = new octree_slam::rendering::CUDARenderer(OCTREE, path_prefix, width, height);
	} else if (DRAW_POINT_CLOUD || (!USE_CUDA_RASTERIZER && !DRAW_CAMERA_COLOR)) {
    gl_renderer_ = new octree_slam::rendering::OpenGLRenderer(path_prefix);
	}

	return true;
}

void shut_down(int return_code){
  cudaFree(points_);
	cudaDeviceReset();
#ifdef __APPLE__
	glfwTerminate();
#endif
	exit(return_code);
}

void errorCallback(int error, const char* description){
	fputs(description, stderr);
}
