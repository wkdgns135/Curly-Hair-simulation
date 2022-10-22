#pragma once
#include <stdio.h>
#include <time.h>
#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <filesystem>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <sys/stat.h>

// added by jhkim
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "controls.hpp"
#include "shader.hpp"
#include "texture.hpp"
#include "objloader.hpp"

using namespace std;

vector<GLfloat> vertex;
vector<GLfloat> vertex_color;
vector<GLfloat> vertex_noise;
vector<glm::vec3> vertex_tangent;
vector<vector<glm::vec3>> vertex2D;

string vertexshader_fn = "SimpleFragmentShader.vertexshader";
string fragmentshader_fn = "SimpleFragmentShader.fragmentshader";
GLFWwindow* window;
// added by jhkim
bool isCapture = false;

vector<string> file_names;
int file_index = 0;

struct path_leaf_string
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		return entry.path().string();
	}
};

void read_directory(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string());
}

void read_hair_asc(const char *filename) {
	FILE *f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename);
		return;
	}
	vertex2D.clear();

	int nstrands = 0;
	if (!fscanf(f, "%d", &nstrands)) {
		fprintf(stderr, "Couldn't read number of strands\n");
		fclose(f);
		return;
	}
	for (int i = 0; i < nstrands; i++) {
		int nverts = 0;
		vector<glm::vec3> v;
		fscanf(f, "%d", &nverts);
		for (int j = 0; j < nverts; j++) {
			glm::vec3 vert;
			if (!fscanf(f, "%f%f%f", &vert.x, &vert.y, &vert.z)) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(f);
				return;
			}
			v.push_back(vert);
		}
		vertex2D.push_back(v);
	}
	fclose(f);
	return;
}


// added by jhkim
void FlipVertically(int width, int height, char *data)
{
	char rgb[3];
	for (int y = 0; y < height / 2; ++y) {
		for (int x = 0; x < width; ++x) {
			int top = (x + y * width) * 3;
			int bottom = (x + (height - y - 1) * width) * 3;
			memcpy(rgb, data + top, sizeof(rgb));
			memcpy(data + top, data + bottom, sizeof(rgb));
			memcpy(data + bottom, rgb, sizeof(rgb));
		}
	}
}

void Capture(void)
{
	static int frame = 0;
	if (frame == 0 || frame % 5 == 0) {
		static int index = 0;
		char filename[100];
		sprintf_s(filename, "capture\\capture-%d.bmp", index);
		GLint viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);

		int x = viewport[0];
		int y = viewport[1];
		int width = viewport[2];
		int height = viewport[3];
		char *data = (char*)malloc((size_t)(width * height * 3)); // 3 components (R, G, B)
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
		FlipVertically(width, height, data);
		stbi_write_bmp(filename, width, height, 3, data);
		free(data);
		index++;
	}
	frame++;
}
void update_vertex() {
	int cnt = 0;
	glm::vec3 point; // previous point

	srand(2210U); // just random seed
	vertex.clear();
	vertex_color.clear();
	vertex_noise.clear();
	vertex_tangent.clear();

	for (int i = 0; i < vertex2D.size(); i++) {
		cnt++;
		vertex.push_back(vertex2D[i][0].x);
		vertex.push_back(vertex2D[i][0].y);
		vertex.push_back(vertex2D[i][0].z);

		//vertex_color.push_back((float)cnt);

		point = glm::vec3(vertex2D[i][0].x, vertex2D[i][0].y, vertex2D[i][0].z); // initial point
		vertex_tangent.push_back(glm::vec3(0, 0, 1)); // tangent is (0, 0, 1) on initial point.
		vertex_noise.push_back(rand());
		for (size_t j = 1; j < vertex2D[i].size(); ++j) {
			vertex.push_back(vertex2D[i][j].x);
			vertex.push_back(vertex2D[i][j].y);
			vertex.push_back(vertex2D[i][j].z);

			//vertex_color.push_back((float)cnt);

			vertex_tangent.push_back(glm::normalize(point - glm::vec3(vertex2D[i][j].x, vertex2D[i][j].y, vertex2D[i][j].z))); // tangent vector

			vertex_noise.push_back(rand());
			if (j < vertex2D[i].size() - 1)
			{
				vertex.push_back(vertex2D[i][j].x);
				vertex.push_back(vertex2D[i][j].y);
				vertex.push_back(vertex2D[i][j].z);

				//vertex_color.push_back((float)cnt);
				vertex_tangent.push_back(glm::normalize(point - glm::vec3(vertex2D[i][j].x, vertex2D[i][j].y, vertex2D[i][j].z))); // tangent vector
				point = glm::vec3(vertex2D[i][j].x, vertex2D[i][j].y, vertex2D[i][j].z); // update previous point  
				vertex_noise.push_back(rand());

				//glm::vec3 t = vertex_tangent.at(vertex_tangent.size()-1);
				//std::cout << t.x << t.y << t.z << std::endl;
			}
		}
	}
}


void render(const char* objpath)
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

	int winsize[2] = { 1440 * 0.7, 1080 * 0.7 }; // added by jhkim

	window = glfwCreateWindow(winsize[0], winsize[1], "Geometric Image Viewer", NULL, NULL); // Windowed
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
	glfwSetCursorPos(window, winsize[0] / 2, winsize[1] / 2);

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
	glm::vec3 eye_pos = glm::vec3(3, 0, 1.3);
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

	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals; // Won't be used at the moment.

	int hair_data_length = vertex.size();
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
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices, GL_STATIC_DRAW);

	GLuint face_uvbuffer;
	glGenBuffers(1, &face_uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, face_uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs, GL_STATIC_DRAW);

	GLuint face_normalbuffer;
	glGenBuffers(1, &face_normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, face_normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals, GL_STATIC_DRAW);

	//std::vector<float> head_color(vertices.size(), 0.0);
	//vertex_color.insert(vertex_color.end(), head_color.begin(), head_color.end());
	bool first = true;
	do {
		// Clear the screen. 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		read_hair_asc(file_names[file_index++ % file_names.size()].c_str());
		update_vertex();


		glUseProgram(programID);
		computeMatricesFromInputs();

		ProjectionMatrix = getProjectionMatrix();
		ViewMatrix = getViewMatrix();
		MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glm::vec3 lightPos = glm::vec3(0, 1, 3);
		glUniform3f(LightID_hair, lightPos.x, lightPos.y, lightPos.z);

		//update buffer
		GLfloat* g_vertex_buffer_data = vertex.data();
		GLuint vertexbuffer;
		glGenBuffers(1, &vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, vertex.size() * sizeof(float), g_vertex_buffer_data, GL_STATIC_DRAW);

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

		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void *)0            // array buffer offset
		);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(
			1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void *)0            // array buffer offset
		);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, noisebuffer);
		glVertexAttribPointer(
			2,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			1,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void *)0            // array buffer offset
		);
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, tangentbuffer);
		glVertexAttribPointer(
			3,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
		glEnableVertexAttribArray(3);

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
		if (isCapture) {
			Capture();
		}
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


int main(int argc, char** argv) {
	const char* name = "head_model";
	read_directory("frame\\", file_names);
	render(name);
}