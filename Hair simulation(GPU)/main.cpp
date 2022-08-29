#pragma once
#include <stdio.h>
#include <time.h>
#include "GL/glut.h"

#include "HairModel.h"

using namespace std;

double zoom = 17.5f;
double rot_x = 90.0f;
double rot_y = 180.0f;
double trans_x = 0.0f;
double trans_y = 0.0f;

int last_x = 0;
int last_y = 0;
unsigned char buttons[3] = { 0 };
double dt = 0.01;

// 0 : bounsing test	Key(B)
// 1 : wind test		Key(W)
// 2 : simulation		key(SPACE)
bool status[3] = { false, false, true}; 
double n = 0;

HairModel *hm;

void Draw() {
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	hm->draw_point();
	hm->draw_wire();
	//hm->draw_frame();
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
		zoom -= (double)0.02f * diff_x;
	}
	else if (buttons[1]) {
		trans_x += (double)0.02f * diff_x;
		trans_y -= (double)0.02f * diff_y;
	}
	else if (buttons[0]) {
		rot_x += (double)0.2f * diff_y;
		rot_y += (double)0.2f *diff_x;
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
	rot_x = 15;
	rot_y = 0;
	glEnable(GL_DEPTH_TEST);
}

void Reshape(int w, int h) {
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, double(w) / h, dt, 1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

void SpecialKeys(int key, int x, int y) {
	switch (key)
	{
	case GLUT_KEY_RIGHT:
		//hm->move_root_particle(Vector3d(0, 0.0, -0.1));
		break;
	case GLUT_KEY_LEFT:
		//hm->move_root_particle(Vector3d(0, 0.0, 0.1));
		break;
	case GLUT_KEY_UP:
		//hm->move_root_particle(Vector3d(0.0, .1, .0));
		break;
	case GLUT_KEY_DOWN:
		//hm->move_root_particle(Vector3d(0.0, -.1, .0));
		break;
	}
	::glutPostRedisplay();
}

void KeyboardEvent(unsigned char key, int x, int y) {
	switch (key)
	{
	case'r':
	case'R':
		//hm->init(hm->particle);
		n = 0;
		break;
	case 'b':
	case 'B':
		status[0] = !status[0];
		break;
	case 'w':
	case 'W':
		status[1] = !status[1];
		break;
	case ' ':
		status[2] = !status[2];
		break;
	case ']':
		//hm->w_c += 1;
		//cout << "wet_c :" << hm->w_c << endl;
		break;
	case '[':
		//hm->w_c -= 1;
		//cout << "wet_c :" << hm->w_c << endl;
		break;
	case 'q':
	case 'Q':
		rot_y += 90;
		break;
	case 'v':
	case 'V':
		Init();
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