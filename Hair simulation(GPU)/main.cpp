#include <nanogui/opengl.h>
#include <nanogui/glutil.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/checkbox.h>
#include <nanogui/button.h>
#include <nanogui/toolbutton.h>
#include <nanogui/popupbutton.h>
#include <nanogui/combobox.h>
#include <nanogui/progressbar.h>
#include <nanogui/entypo.h>
#include <nanogui/messagedialog.h>
#include <nanogui/textbox.h>
#include <nanogui/slider.h>
#include <nanogui/imagepanel.h>
#include <nanogui/imageview.h>
#include <nanogui/vscrollpanel.h>
#include <nanogui/colorwheel.h>
#include <nanogui/graph.h>
#include <nanogui/tabwidget.h>
#include <nanogui/glcanvas.h>
#include <iostream>
#include <string>

#include "HairModel.h"

// Includes for the GLTexture class.
#include <cstdint>
#include <memory>
#include <utility>

#if defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#if defined(_WIN32)
#  pragma warning(push)
#  pragma warning(disable: 4457 4456 4005 4312)
#endif

#if defined(_WIN32)
#  pragma warning(pop)
#endif
#if defined(_WIN32)
#  if defined(APIENTRY)
#    undef APIENTRY
#  endif
#  include <windows.h>
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::to_string;

HairModel *hm;


class SimulationCanvas : public nanogui::GLCanvas {
public:
	SimulationCanvas(Widget *parent) : nanogui::GLCanvas(parent), mRotation(nanogui::Vector3f(0.25f, 0.5f, 0.33f)) {
		using namespace nanogui;
		hm = new HairModel();

		mShader.init(
			/* An identifying name */
			"a_simple_shader",

			/* Vertex shader */
			"#version 330\n"
			"uniform mat4 modelViewProj;\n"
			"in vec3 position;\n"
			"in vec3 color;\n"
			"out vec4 frag_color;\n"
			"void main() {\n"
			"    frag_color = vec4(color, 1.0);\n"
			"    gl_Position = modelViewProj * vec4(position, 1.0);\n"
			"}",

			/* Fragment shader */
			"#version 330\n"
			"out vec4 color;\n"
			"in vec4 frag_color;\n"
			"void main() {\n"
			"    color = frag_color;\n"
			"}"
		);

		MatrixXu indices(2, hm->TOTAL_SIZE); /* Draw a cube */
		setIndex(indices);

		MatrixXf positions(3, hm->TOTAL_SIZE);
		setPosition(hm->particle_host.position, positions);

		hm->get_colors();
		MatrixXf colors(3, hm->TOTAL_SIZE);
		setColor(hm->colors, colors);

		mShader.bind();
		mShader.uploadIndices(indices);

		mShader.uploadAttrib("position", positions);
		mShader.uploadAttrib("color", colors);
	}

	~SimulationCanvas() {
		mShader.free();
	}

	
	void setIndex(nanogui::MatrixXu &mat) {
		for (int i = 0; i < hm->STRAND_SIZE; i++) {
			for (int j = 0; j < hm->MAX_SIZE-1; j++) {
				mat.col(i*hm->MAX_SIZE + j) << i * hm->MAX_SIZE + j, i * hm->MAX_SIZE + j + 1;
			}
		}
	}

	void setPosition(float3 *v, nanogui::MatrixXf &mat) {
		for (int i = 0; i < hm->TOTAL_SIZE; i++) {

			mat.col(i) << v[i].x * 0.02, v[i].y* 0.02, v[i].z* 0.02;
		}
	}

	void setColor(float3 *v, nanogui::MatrixXf &mat) {
		for (int i = 0; i < hm->STRAND_SIZE; i++) {
			for (int j = 0; j < hm->MAX_SIZE - 1; j++) {
				mat.col(i*hm->MAX_SIZE + j) << v[i*hm->MAX_SIZE + j].x, v[i*hm->MAX_SIZE + j].y, v[i*hm->MAX_SIZE + j].z;
			}
		}
	}

	void setRotation(nanogui::Vector3f vRotation) {
		mRotation = vRotation;
	}

	virtual void drawGL() override {
		using namespace nanogui;

		hm->simulation();
		hm->get_colors();

		MatrixXu indices(2, hm->TOTAL_SIZE); /* Draw a cube */
		setIndex(indices);

		MatrixXf positions(3, hm->TOTAL_SIZE);
		setPosition(hm->particle_host.position, positions);

		hm->get_colors();
		MatrixXf colors(3, hm->TOTAL_SIZE);
		setColor(hm->colors, colors);

		mShader.bind();
		mShader.uploadIndices(indices);

		mShader.uploadAttrib("position", positions);
		mShader.uploadAttrib("color", colors);
		mShader.bind();


		Matrix4f mvp;
		mvp.setIdentity();
		//float fTime = (float)glfwGetTime();
		//mvp.topLeftCorner<3, 3>() = Eigen::Matrix3f(Eigen::AngleAxisf(mRotation[0] * fTime, Vector3f::UnitX()) *
		//	Eigen::AngleAxisf(mRotation[1] * fTime, Vector3f::UnitY()) *
		//	Eigen::AngleAxisf(mRotation[2] * fTime, Vector3f::UnitZ())) * 0.25f;

		mShader.setUniform("modelViewProj", mvp);

		glEnable(GL_DEPTH_TEST);
		/* Draw 12 triangles starting at index 0 */
		mShader.drawIndexed(GL_LINES, 0, hm->TOTAL_SIZE);
		glDisable(GL_DEPTH_TEST);
	}

private:
	nanogui::GLShader mShader;
	Eigen::Vector3f mRotation;
};


class MainScene : public nanogui::Screen {\
public:
	MainScene() : nanogui::Screen(Eigen::Vector2i(800, 600), "MainScene", false, true) {
		using namespace nanogui;
		//Simuation window
		Window *window = new Window(this, "Simulation window");
		window->setPosition(Vector2i(15, 15));
		window->setLayout(new GroupLayout());
		
		simulation_canvas = new SimulationCanvas(window);
		simulation_canvas->setBackgroundColor({ 100, 100, 100, 255 });
		simulation_canvas->setSize({ 400, 400 });

		//Parameter window
		window = new Window(this, "Grid of small widgets");
		window->setPosition(Vector2i(425, 300));
		GridLayout *layout =
			new GridLayout(Orientation::Horizontal, 2,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
			{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		window->setLayout(layout);

		new Label(window, "Test slider", "sans-bold");
		Widget *tools = new Widget(window);
		tools->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		Slider *slider = new Slider(tools);
		slider->setValue(0.5f);
		slider->setFixedWidth(160);
		slider->setFinalCallback([&](float value) {
			cout << "test1:" << value << endl;
		});

		new Label(window, "Test slider2", "sans-bold");
		tools = new Widget(window);
		tools->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		slider = new Slider(tools);
		slider->setValue(0.5f);
		slider->setFixedWidth(160);
		slider->setFinalCallback([&](float value) {
			cout << "test2:" << value << endl;
		});
		
		performLayout();
	}
	virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) {
		if (Screen::keyboardEvent(key, scancode, action, modifiers))
			return true;
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			setVisible(false);
			return true;
		}
		return false;
	}

	virtual void draw(NVGcontext *ctx) {
		/* Draw the user interface */
		Screen::draw(ctx);
	}
private:
	SimulationCanvas *simulation_canvas;
};

int main(int /* argc */, char ** /* argv */) {
	try {
		nanogui::init();

		/* scoped variables */ {
			nanogui::ref<MainScene> app = new MainScene();
			app->drawAll();
			app->setVisible(true);
			nanogui::mainloop();
		}

		nanogui::shutdown();
	}
	catch (const std::runtime_error &e) {
		std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
#if defined(_WIN32)
		MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
#else
		std::cerr << error_msg << endl;
#endif
		return -1;
	}

	return 0;
}

//#pragma once
//#include <stdio.h>
//#include <time.h>
//#include "GL/glut.h"
//#include "HairModel.h"
//#include "FileController.h"
//using namespace std;
//
//float zoom = 17.5f;
//float rot_x = 90.0f;
//float rot_y = 180.0f;
//float trans_x = 0.0f;
//float trans_y = 0.0f;
//
//int last_x = 0;
//int last_y = 0;
//unsigned char buttons[3] = { 0 };
//double dt = 0.01;
//
//// 0 : bounsing test	Key(B)
//// 1 : wind test		Key(W)
//// 2 : simulation		key(SPACE)
//bool status[4] = { false, false, true, false };
//double n = 0;
//
//bool out_file = false;
//int out_file_num = 0;
//
//bool out_capture = false;
//
//HairModel *hm;
//void Draw() {
//	glEnable(GL_LIGHTING);
//	glEnable(GL_LIGHT0);
//
//
//	glPushMatrix();
//	//glTranslatef(hm->sphere[0], hm->sphere[1], hm->sphere[2]);
//	//glutSolidSphere(hm->radius - 0.01, 20, 20);
//	glPopMatrix();
//
//	//hm->draw_point();
//	hm->draw_wire();
//	//hm->test_draw();
//
//	glDisable(GL_LIGHTING);
//}
//
//
//void Mouse(int button, int state, int x, int y) {
//	last_x = x;
//	last_y = y;
//
//	switch (button)
//	{
//	case GLUT_LEFT_BUTTON:
//		buttons[0] = state == GLUT_DOWN ? 1 : 0;
//		break;
//	case GLUT_MIDDLE_BUTTON:
//		buttons[1] = state == GLUT_DOWN ? 1 : 0;
//		break;
//	case GLUT_RIGHT_BUTTON:
//		buttons[2] = state == GLUT_DOWN ? 1 : 0;
//		break;
//
//	default:
//		break;
//	}
//
//	glutPostRedisplay();
//}
//void Motion(int x, int y) {
//	int diff_x = x - last_x;
//	int diff_y = y - last_y;
//	last_x = x;
//	last_y = y;
//
//	if (buttons[2]) {
//		zoom -= (float)0.02f * diff_x;
//	}
//	else if (buttons[1]) {
//		trans_x += (float)0.02f * diff_x;
//		trans_y -= (float)0.02f * diff_y;
//	}
//	else if (buttons[0]) {
//		rot_x += (float)0.2f * diff_y;
//		rot_y += (float)0.2f *diff_x;
//	}
//
//	glutPostRedisplay();
//}
//
//
//void Display(void) {
//	//glClearColor(0.6, 0.6, 0.6, 1);
//	glClearColor(0, 0, 0, 1);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glLoadIdentity();
//
//	glTranslatef(0, 0, -zoom);
//	glTranslatef(trans_x, trans_y, 0);
//	glRotatef(rot_x, 1, 0, 0);
//	glRotatef(rot_y, 0, 1, 0);
//	Draw();
//	glutSwapBuffers();
//}
//
//void Init(void) {
//	zoom = 20;
//	trans_x = 0;
//	trans_y = zoom / 3;
//	rot_x = 15;
//	rot_y = 0;
//
//	glEnable(GL_DEPTH_TEST);
//}
//
//void Reshape(int w, int h) {
//	if (w == 0) {
//		h = 1;
//	}
//	glViewport(0, 0, w, h);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	gluPerspective(45.0, float(w) / h, dt, 1000);
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//
//}
//
//void SpecialKeys(int key, int x, int y) {
//	switch (key)
//	{
//	case GLUT_KEY_RIGHT:
//		break;
//	case GLUT_KEY_LEFT:
//		break;
//	case GLUT_KEY_UP:
//		break;
//	case GLUT_KEY_DOWN:
//		break;
//	}
//	::glutPostRedisplay();
//}
//
//void KeyboardEvent(unsigned char key, int x, int y) {
//	switch (key)
//	{
//	case'r':
//	case'R':
//		break;
//	case 'b':
//	case 'B':
//		status[0] = !status[0];
//		break;
//	case 'w':
//	case 'W':
//		status[1] = !status[1];
//		break;
//	case ' ':
//		status[2] = !status[2];
//		break;
//	case 'S':
//	case 's':
//		status[3] = !status[3];
//		break;
//	case 'q':
//	case 'Q':
//		rot_y += 90;
//		break;
//	case 'v':
//	case 'V':
//		Init();
//		break;
//	default:
//		break;
//	}
//}
//
//void upLinePrompt(int count)
//{
//	for (int i = 0; i < count; ++i) {
//		//printf("%c[2K",27);
//		cout << "\33[2K"; //line clear
//		cout << "\x1b[A"; //up line (ESC [ A) must be support VT100 escape seq
//	}
//}
//
//void Update() {
//	hm->simulation();
//
//	if (out_file)out_hair_asc(hm, "Test", out_file_num++);
//	::glutPostRedisplay();
//}
//
//int main(int argc, char** argv) {
//	hm = new HairModel();
//	glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
//	glutInitWindowSize(680, 680);
//	glutInitWindowPosition(100, 100);
//
//	glutCreateWindow("Hair Simulation");
//	glutDisplayFunc(Display);
//	glutReshapeFunc(Reshape);
//	glutMouseFunc(Mouse);
//
//	glutMotionFunc(Motion);
//	glutSpecialFunc(SpecialKeys);
//	glutKeyboardFunc(KeyboardEvent);
//	glutIdleFunc(Update);
//
//	Init();
//	glutMainLoop();
//}