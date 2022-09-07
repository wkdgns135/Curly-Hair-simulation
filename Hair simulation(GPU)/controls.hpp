#ifndef CONTROLS_HPP
#define CONTROLS_HPP
#include "HairModel.h"

void computeMatricesFromInputs(HairModel *hm);
glm::mat4 getViewMatrix();
glm::mat4 getProjectionMatrix();
glm::vec3 getCameraPosition();
int getNormalDisplayMode();
bool getTextureMode();

void setTarget(glm::highp_f64vec3 obj_target);

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

#endif