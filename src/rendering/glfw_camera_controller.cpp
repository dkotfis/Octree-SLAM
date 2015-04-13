
// Octree-SLAM Dependency
#include <octree_slam/rendering/glfw_camera_controller.h>

// OpenGL Dependency
#include <glm/gtc/matrix_transform.hpp>

namespace octree_slam {

namespace rendering {

//Initialize static member variables
float GLFWCameraController::FoV_ = 45.0f;
bool GLFWCameraController::LB_ = false;

GLFWCameraController::GLFWCameraController(GLFWwindow *window, const int width, const int height) :
  position_(0.0f, 0.0f, 0.0f), 
  horizontalAngle_(0.0f), 
  verticalAngle_(0.0f), 
  zNear_(0.001f), 
  zFar_(10000.0f), 
  speed_(1.0f), 
  mouseSpeed_(0.005f), 
  lastTime_(glfwGetTime()),
  width_(width),
  height_(height),
  window_(window) {

  glfwSetMouseButtonCallback(window_, mouseClickCallback);
  glfwSetScrollCallback(window_, scrollCallback);
}

GLFWCameraController::~GLFWCameraController() {


}

void GLFWCameraController::update() {

  //Get latest events
  glfwPollEvents();

  //Compute the current time
  double currentTime = glfwGetTime();
  float deltaTime = float(currentTime = lastTime_);
  lastTime_ = currentTime;

  //Only use the mouse if the left button is clicked
  if (LB_) {
    //Read the mouse position
    double xpos, ypos;
    glfwGetCursorPos(window_, &xpos, &ypos);

    //Reset the mouse position to the center
    glfwSetCursorPos(window_, width_ / 2, height_ / 2);

    //Compute the viewing angle
    horizontalAngle_ += mouseSpeed_ * deltaTime * float(width_ / 2 - xpos);
    verticalAngle_ += mouseSpeed_ * deltaTime * float(height_ / 2 - ypos);
  }

  //Compute the direction
  glm::vec3 direction(cos(verticalAngle_) * sin(horizontalAngle_), sin(verticalAngle_), cos(verticalAngle_)*cos(horizontalAngle_));
  glm::vec3 right = glm::vec3(sin(horizontalAngle_ - 3.14f / 2.0f), 0, cos(horizontalAngle_ - 3.14f / 2.0f));
  glm::vec3 up = glm::cross(right, direction);

  //Compute position from keyboard inputs
  if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS) {
    position_ += direction * deltaTime * speed_;
  }
  if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS) {
    position_ -= direction * deltaTime * speed_;
  }
  if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) {
    position_ += right * deltaTime * speed_;
  }
  if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS) {
    position_ -= right * deltaTime * speed_;
  }

  //Update the matrices
  camera_.model = glm::mat4(1.0f);
  camera_.view = glm::lookAt(position_, position_ + direction, up);
  camera_.projection = glm::perspective(FoV_, (float)(width_) / (float)(height_), zNear_, zFar_);
  camera_.modelview = camera_.view * camera_.model;
  camera_.mvp = camera_.projection*camera_.modelview;
  camera_.fov = FoV_;

}

void GLFWCameraController::mouseClickCallback(GLFWwindow *window, int button, int action, int mods){
  if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
    LB_ = true;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  }

  if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT) {
    LB_ = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  }
}

void GLFWCameraController::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  //Update the field of view from the mouse scroll wheel
  FoV_ -= 2 * yoffset;
}

} // namespace rendering

} // namespace octree_slam
