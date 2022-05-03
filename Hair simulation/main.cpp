#pragma once
#include <stdio.h>
#include <time.h>
#include "GL/glut.h"
#include "HairModel.h"
using namespace std;

float zoom = 17.5f;
float rot_x = 90.0f;
float rot_y = 180.0f;
float trans_x = 0.0f;
float trans_y = 0.0f;

int last_x = 0;
int last_y = 0;
unsigned char buttons[3] = { 0 };
double dt = 0.01;

HairModel *hm;

#pragma region Draw section
void posed_curves() {
	glColor3f(0, 0, 0);
	hm->draw_point(hm->particle->pos);
	hm->draw_wire(hm->particle->pos);
}

void smoothed_posed_curves() {
	glColor3f(0, 0, 1);
	hm->draw_point(hm->smoothed_particle->pos);
	hm->draw_wire(hm->smoothed_particle->pos);
}

void rest_curves() {
	glColor3f(0, 0, 1);
	hm->draw_point(hm->rest_particle->pos);
	hm->draw_wire(hm->rest_particle->pos);
}

void smoothed_rest_curves() {
	glColor3f(1, 0, 0);
	hm->draw_point(hm->smoothed_rest_particle->pos);
	hm->draw_wire(hm->smoothed_rest_particle->pos);
}

#pragma endregion

void Draw() {
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	//posed_curves();
	//smoothed_posed_curves();
	//hm->draw_frame(hm->smoothed_particle);

	//hm->draw_frame(hm->smoothed_rest_particle);
	rest_curves();
	smoothed_rest_curves();

	glDisable(GL_LIGHTING);
}

void Mouse(int button, int state, int x, int y) {
	last_x = x;
	last_y = y;

	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		buttons[0] = state == GLUT_DOWN ? 1 : 0;
		break;
	case GLUT_MIDDLE_BUTTON:
		buttons[1] = state == GLUT_DOWN ? 1 : 0;
		break;
	case GLUT_RIGHT_BUTTON:
		buttons[2] = state == GLUT_DOWN ? 1 : 0;
		break;

	default:
		break;
	}

	glutPostRedisplay();
}
void Motion(int x, int y) {
	int diff_x = x - last_x;
	int diff_y = y - last_y;
	last_x = x;
	last_y = y;

	if (buttons[2]) {
		zoom -= (float)0.02f * diff_x;
	}
	else if (buttons[1]) {
		trans_x += (float)0.02f * diff_x;
		trans_y -= (float)0.02f * diff_y;
	}
	else if (buttons[0]) {
		rot_x += (float)0.2f * diff_y;
		rot_y += (float)0.2f *diff_x;
	}

	glutPostRedisplay();
}


void Display(void) {
	glClearColor(0.6, 0.6, 0.6, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glTranslatef(0, 0, -zoom);
	glTranslatef(trans_x, trans_y, 0);
	glRotatef(rot_x, 1, 0, 0);
	glRotatef(rot_y, 0, 1, 0);
	Draw();
	glutSwapBuffers();
}

void Init(void) {
	zoom = 20;
	trans_x = 0;
	trans_y = zoom / 3;
	rot_x = 45;
	rot_y = -90;
	glEnable(GL_DEPTH_TEST);
}

void Reshape(int w, int h) {
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, float(w) / h, dt, 1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

void SpecialKeys(int key, int x, int y) {
	switch (key)
	{
	case GLUT_KEY_RIGHT:
		hm->move_root_particle(Vector3d(0, 0.0, -0.1));
		break;
	case GLUT_KEY_LEFT:
		hm->move_root_particle(Vector3d(0, 0.0, 0.1));
		break;
	case GLUT_KEY_UP:
		hm->move_root_particle(Vector3d(0.0, .1, .0));
		break;
	case GLUT_KEY_DOWN:
		hm->move_root_particle(Vector3d(0.0, -.1, .0));
		break;
	}
	::glutPostRedisplay();
}

void KeyboardEvent(unsigned char key, int x, int y) {
	switch (key)
	{
	case'r':
	case'R':
		hm->init(hm->particle);
		break;
	case 't':
	case 'T':

		break;
	default:
		break;
	}
}

void upLinePrompt(int count)
{
	for (int i = 0; i < count; ++i) {
		//printf("%c[2K",27);
		cout << "\33[2K"; //line clear
		cout << "\x1b[A"; //up line (ESC [ A) must be support VT100 escape seq
	}
}

void Update() {
	hm->simulation();
	::glutPostRedisplay();
}

int main(int argc, char** argv) {
	hm = new HairModel();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(680, 680);
	glutInitWindowPosition(100, 100);

	glutCreateWindow("Hair Simulation");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutMouseFunc(Mouse);

	glutMotionFunc(Motion);
	glutSpecialFunc(SpecialKeys);
	glutKeyboardFunc(KeyboardEvent);
	glutIdleFunc(Update);

	Init();
	glutMainLoop();
}