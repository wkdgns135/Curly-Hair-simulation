// Include GLFW
#include <GLFW/glfw3.h>
extern GLFWwindow* window; 

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;
#include <stdio.h>
#include "controls.hpp"
#include "vector_functions.h"

glm::mat4 ViewMatrix;
glm::mat4 ProjectionMatrix;

glm::mat4 getViewMatrix(){
	return ViewMatrix;
}
glm::mat4 getProjectionMatrix(){
	return ProjectionMatrix;
}

//float initialRadius = 1.3;
//float currentRadius = 1.3;
float initialRadius = -238.0; // modified by jhkim
float currentRadius = -238.0; // modified by jhkim

int normalMode = 0; // 0 : normal, 1: fine normal;
bool textureMode = false;

// Initial position : on +Z
glm::vec3 position = glm::vec3(0, 0, 1.3);
glm::vec3 target = glm::vec3(0.0, -20.0, -1);

// Initial horizontal angle : toward -Z
float horizontalAngle = 0.0f;//3.14f;
// Initial vertical angle : upside down
float verticalAngle = 0.0f;
// Initial Field of View
float initialFoV = 26.5f;


float speed = 100.0f; // 2 units / second
float mouseSpeed = 0.005f;

bool getTextureMode()
{
	return textureMode;
}

int getNormalDisplayMode()
{
	return normalMode;
}

glm::vec3 getCameraPosition()
{
	return position;
}

void setTarget(glm::highp_f64vec3 obj_target)
{
	target.x = (glm::f32)obj_target.x;
	target.y = (glm::f32)obj_target.y;
	target.z = (glm::f32)obj_target.z;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	currentRadius += yoffset * 5;
}


double prev_xpos, prev_ypos;
bool isMousePressed = false;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if(action == GLFW_PRESS){
            isMousePressed = true;
			glfwGetCursorPos(window, &prev_xpos, &prev_ypos);
		
		}
        else if(action == GLFW_RELEASE)
            isMousePressed = false;		
    }
}

void computeMatricesFromInputs(HairModel *hm)
{

    if(isMousePressed) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		horizontalAngle += mouseSpeed * float(prev_xpos - xpos );
		verticalAngle   -= mouseSpeed * float( prev_ypos - ypos );

		prev_xpos = xpos;
		prev_ypos = ypos;
    }
	else{
		glfwGetCursorPos(window, &prev_xpos, &prev_ypos);
	}

	// Move forward
	if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS){
		hm->move_sphere(make_float3(0, 1.0f, 0));
	}
	// Move backward
	if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS){
		hm->move_sphere(make_float3(0, -1.0f, 0));

	}
	// Strafe right
	if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
		hm->move_sphere(make_float3(0, 0, 1.0f));
		//normalMode = 1;
	}
	// Strafe left
	if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS){
		hm->move_sphere(make_float3(0, 0, -1.0f));
		//normalMode = 0;
	}
	// T key
	if(glfwGetKey( window, GLFW_KEY_T) == GLFW_PRESS){
		textureMode = true;
	}
	// R key
	if(glfwGetKey( window, GLFW_KEY_R) == GLFW_PRESS){
		textureMode = false;
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		hm->move_root(0);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		hm->move_root(1);
	}

	glm::mat3x3 rotmat_y(cos(horizontalAngle), 0.0, sin(horizontalAngle), 0.0, 1.0, 0.0, -sin(horizontalAngle), 0.0, cos(horizontalAngle));
	glm::mat3x3 rotmat_x(1.0, 0.0, 0.0, 0.0, cos(verticalAngle), -sin(verticalAngle), 0.0, sin(verticalAngle), cos(verticalAngle));
	glm::mat3x3 rotmat = rotmat_x * rotmat_y;

	position = rotmat * (glm::vec3(0.0, 0.0, currentRadius) - target) + target;
	glm::vec3 up = rotmat * glm::vec3(0.0, 1.0, 0.0);

	float FoV = initialFoV;

	// Projection matrix : 45� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	ProjectionMatrix = glm::perspective(glm::radians(26.5f), float(6.0f / 6.0f), 0.1f, 2000.0f);
	// Camera matrix
	ViewMatrix       = glm::lookAt(
								position,           // Camera is here
								target, //position+direction, // and looks here : at the same position, plus "direction"
								up                  // Head is up (set to 0,-1,0 to look upside-down)
						   );
}