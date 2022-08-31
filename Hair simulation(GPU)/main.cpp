#pragma once
#include <stdio.h>
#include <time.h>
#include <vector>
#include <string>

#include "HairModel.h"
#include "controls.hpp"
#include "shader.hpp"
#include "texture.hpp"
#include "objloader.hpp"

using namespace std;

HairModel *hm;

std::string vertexshader_fn = "SimpleVertexShader.vertexshader";
std::string fragmentshader_fn = "SimpleFragmentShader.fragmentshader";
GLFWwindow* window;

void render(std::vector<GLfloat> &vertex_data, std::vector<GLfloat> &vertex_color,
	std::vector<GLfloat> &vertex_noise, std::vector<glm::vec3> &vertex_tangent, char* objpath)
{
	glewExperimental = true; // Needed for core profile
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return;
	}

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);


	window = glfwCreateWindow(1440, 1080, "Geometric Image Viewer", NULL, NULL); // Windowed
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return;
	}
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);


	// GLFW callbacks to handle inputs
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// Set the mouse at the center of the screen
	glfwPollEvents();
	glfwSetCursorPos(window, 1440 / 2, 1080 / 2);

	glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	glEnable(GL_CULL_FACE);

	//VAO
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	GLuint programID = LoadShaders("SimpleVertexShader.vertexshader", "SimpleFragmentShader.fragmentshader");
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(programID);

	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	GLuint LightID_hair = glGetUniformLocation(programID, "LightPosition_worldspace");

	// Draw using a shader

	glm::mat4 ProjectionMatrix = glm::perspective(glm::radians(26.5f), (float)(6.0 / 6.0), 0.5f, 100.0f);
	glm::vec3 eye_pos = glm::vec3(0, 0, 1.3);
	glm::vec3 look_pos = glm::vec3(0, 0, -1);
	glm::vec3 head_up = glm::vec3(0, -1, 0);

	// Camera matrix
	glm::mat4 ViewMatrix = glm::lookAt(
		eye_pos, // Camera is at (4,3,3), in World Space
		look_pos, // and looks at the origin
		head_up // Head is up (set to 0,-1,0 to look upside-down)
	);

	//computeMatricesFromInputs();
	//glm::mat4 ProjectionMatrix = getProjectionMatrix();
	//glm::mat4 ViewMatrix = getViewMatrix();

	glm::mat4 ModelMatrix = glm::translate(glm::vec3(0, 0, 0)) * glm::orientate4(glm::vec3(0, 0, 0)) * glm::translate(glm::vec3(0, -1.7, 0));

	glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
	glm::vec3 cam_pos = getCameraPosition();

	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
	GLfloat* g_vertex_buffer_data = vertex_data.data();
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertex_data.size() * sizeof(float), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLfloat* g_vertex_color_data = vertex_color.data();
	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertex_color.size() * sizeof(float), g_vertex_color_data, GL_STATIC_DRAW);

	GLfloat* g_vertex_noise_data = vertex_color.data();
	GLuint noisebuffer;
	glGenBuffers(1, &noisebuffer);
	glBindBuffer(GL_ARRAY_BUFFER, noisebuffer);
	glBufferData(GL_ARRAY_BUFFER, vertex_noise.size() * sizeof(float), g_vertex_noise_data, GL_STATIC_DRAW);

	GLuint tangentbuffer;
	glGenBuffers(1, &tangentbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, tangentbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertex_tangent.size() * sizeof(glm::vec3), &vertex_tangent[0], GL_STATIC_DRAW);

	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals; // Won't be used at the moment.

	int hair_data_length = vertex_data.size();
	// store data length of hair

	std::string name(objpath);
	bool res = loadOBJ((name + ".obj").c_str(), vertices, uvs, normals);

	GLuint face_programID = LoadShaders("normalShader.vertexshader", "normalShader.fragmentshader");

	// Face Shader :: Get a handle for our "MVP" uniform
	glUseProgram(face_programID);
	GLuint face_MatrixID = glGetUniformLocation(face_programID, "MVP");
	GLuint face_ViewMatrixID = glGetUniformLocation(face_programID, "V");
	GLuint face_ModelMatrixID = glGetUniformLocation(face_programID, "M");
	GLuint LightID = glGetUniformLocation(face_programID, "LightPosition_worldspace");

	GLuint NormalTexture = loadBMP_custom((name + ".isomap.bmp").c_str());
	GLuint NormalTextureID = glGetUniformLocation(face_programID, "myTextureSampler");

	GLuint face_vertexbuffer;
	glGenBuffers(1, &face_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, face_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	GLuint face_uvbuffer;
	glGenBuffers(1, &face_uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, face_uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

	GLuint face_normalbuffer;
	glGenBuffers(1, &face_normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, face_normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals[0], GL_STATIC_DRAW);

	//std::vector<float> head_color(vertices.size(), 0.0);
	//vertex_color.insert(vertex_color.end(), head_color.begin(), head_color.end());
	bool first = true;
	do {
		// Clear the screen. 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(programID);

		computeMatricesFromInputs();
		ProjectionMatrix = getProjectionMatrix();
		ViewMatrix = getViewMatrix();
		MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glm::vec3 lightPos = glm::vec3(0, 1, 3);
		glUniform3f(LightID_hair, lightPos.x, lightPos.y, lightPos.z);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void *)0            // array buffer offset
		);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(
			1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			1,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void *)0            // array buffer offset
		);

		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, noisebuffer);
		glVertexAttribPointer(
			2,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			1,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void *)0            // array buffer offset
		);


		glEnableVertexAttribArray(3);
		glBindBuffer(GL_ARRAY_BUFFER, tangentbuffer);
		glVertexAttribPointer(
			3,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		glLineWidth(5.0f);

		glDrawArrays(GL_LINES, 0, hair_data_length / 3);

		glUseProgram(face_programID);

		glUniformMatrix4fv(face_MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(face_ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
		glUniformMatrix4fv(face_ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);

		glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);


		// texture
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, NormalTexture);
		// Set our "myTextureSampler" sampler to use Texture Unit 0
		glUniform1i(NormalTextureID, 0);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, face_vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, face_uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, face_normalbuffer);
		glVertexAttribPointer(
			2,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		glDrawArrays(GL_TRIANGLES, 0, vertices.size());

		//glDrawArrays(GL_TRIANGLES, hair_data_length/3, vertex_data.size()/3);
		//printf("drawing done\n");
		if (first) // save first image only.
		{
			//saveimage(1280, 960);
			first = false;
		}
		glfwSwapBuffers(window);
		glfwPollEvents();

	} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	//saveimage(1280, 960);

	glDeleteVertexArrays(1, &VertexArrayID);
	glfwPollEvents();
	//glDeleteBuffers(1, &vertexbuffer);
	glDeleteProgram(programID);
	//glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

void load_vertex()
{
	int cnt = 0;
	glm::vec3 point; // previous point

	srand(2210U); // just random seed

	for (int i = 0; i < hm->v.size(); i++) {
		cnt++;
		hm->vertex.push_back(hm->p_p[0].x);
		hm->vertex.push_back(hm->p_p[0].y);
		hm->vertex.push_back(hm->p_p[0].z);

		hm->vertex_color.push_back((float)cnt);

		point = glm::vec3(hm->p_p[0].x, hm->p_p[0].y, hm->p_p[0].z); // initial point
		hm->vertex_tangent.push_back(glm::vec3(0, 0, 1)); // tangent is (0, 0, 1) on initial point.
		hm->vertex_noise.push_back(rand());
		for (size_t j = 1; j < hm->v[j].size(); ++i) {
			hm->vertex.push_back(hm->p_p[j].x);
			hm->vertex.push_back(hm->p_p[j].y);
			hm->vertex.push_back(hm->p_p[j].z);

			hm->vertex_color.push_back((float)cnt);

			hm->vertex_tangent.push_back(glm::normalize(point - glm::vec3(hm->p_p[j].x, hm->p_p[j].y, hm->p_p[j].z))); // tangent vector

			hm->vertex_noise.push_back(rand());
			if (i < hm->v[j].size() - 1)
			{
				hm->vertex.push_back(hm->p_p[j].x);
				hm->vertex.push_back(hm->p_p[j].y);
				hm->vertex.push_back(hm->p_p[j].z);

				hm->vertex_color.push_back((float)cnt);
				hm->vertex_tangent.push_back(glm::normalize(point - glm::vec3(hm->p_p[j].x, hm->p_p[j].y, hm->p_p[j].z))); // tangent vector
				point = glm::vec3(hm->p_p[j].x, hm->p_p[j].y, hm->p_p[j].z); // update previous point  
				hm->vertex_noise.push_back(rand());

				//glm::vec3 t = vertex_tangent.at(vertex_tangent.size()-1);
				//std::cout << t.x << t.y << t.z << std::endl;
			}
		}

	}

	for (int i = 0; i < hm->vertex_color.size(); i++) hm->vertex_color[i] /= (float)cnt; // [0,1), same along a strand
}

int main(int argc, char** argv) {
	hm = new HairModel();

	load_vertex();
	render(hm->vertex, hm->vertex_color, hm->vertex_noise, hm->vertex_tangent, "head_model");
}