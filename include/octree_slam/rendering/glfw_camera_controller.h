#ifndef CAMERA_CONTROLLER_H_
#define CAMERA_CONTROLLER_H_

// OpenGL Dependencies
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Octree-SLAM Deependency
#include <octree_slam/scene_structs.h>

namespace octree_slam {

namespace rendering {

class GLFWCameraController {

public:

  GLFWCameraController(GLFWwindow *window, const int width, const int height);

  ~GLFWCameraController();

  void update();

  const Camera camera() const { return camera_; };

private:

  //GLFW input callback functions
  static void mouseClickCallback(GLFWwindow *window, int button, int action, int mods);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

  glm::vec3 position_;
  float horizontalAngle_;
  float verticalAngle_;
  static float FoV_;
  float zNear_;
  float zFar_;
  float speed_;
  float mouseSpeed_;
  double lastTime_;

  //The extents of the screen
  int width_; 
  int height_;

  //Camera matrices
  Camera camera_;

  GLFWwindow* window_;

  static bool LB_;

}; // class GLFWCameraController


} // namespace rendering

} // namespace octree_slam

#endif // CAMERA_CONTROLLER_H_
