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
#include <nanogui/colorpicker.h>
#include <nanogui/graph.h>
#include <nanogui/tabwidget.h>
#include <nanogui/glcanvas.h>
#include <iostream>
#include <string>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "HairModel.h"
#include "vector_calc.h"

// Includes for the GLTexture class.
#include <cstdint>
#include <memory>
#include <utility>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::to_string;

HairModel *hm;
unsigned int num_strands = 128;
bool is_simulation = true;

class SimulationCanvas : public nanogui::GLCanvas {
private:
	nanogui::GLShader hair_shader;
	nanogui::GLShader sphere_shader;
	Eigen::Vector3f mRotation;
	Eigen::Matrix4f ModelMatrix;
	Eigen::Matrix4f ViewMatrix;
	Eigen::Matrix4f ProjectionMatrix;
	
	float initialRadius = -238.0; // modified by jhkim
	float currentRadius = -238.0; // modified by jhkim
	float horizontalAngle = 0.0f;//3.14f;
	float verticalAngle = 0.0f;

public:
	SimulationCanvas(Widget *parent) : nanogui::GLCanvas(parent), mRotation(nanogui::Vector3f(0.25f, 0.5f, 0.33f)) {
		using namespace nanogui;
		hm = new HairModel("strands\\strands00000.txt", num_strands);
		hm->color = make_float3(1.0, 0.8, 0.0);

		hair_shader.init(
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
		
		sphere_shader.init(
			/* an identifying name */
			"a_simple_sphere_shader",

			/* Vertex shader */
			"#version 330\n"
			"uniform mat4 modelViewProj;\n"
			"in vec3 position;\n"
			"out vec4 frag_color;\n"
			"void main() {\n"
			"    frag_color = vec4(0.7, 0.7, 0.7, 1.0);\n"
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
		
		MatrixXu indices(2, hm->TOTAL_SIZE);
		setIndex(indices);

		MatrixXf positions(3, hm->TOTAL_SIZE);
		setPosition(hm->particle_host.position, positions);

		hm->get_colors();
		MatrixXf colors(3, hm->TOTAL_SIZE);
		setColor(hm->colors, colors);

		hair_shader.bind();
		hair_shader.uploadIndices(indices);

		hair_shader.uploadAttrib("position", positions);
		hair_shader.uploadAttrib("color", colors);


		MatrixXf sphere_pos(3, 20402);
		MatrixXu sphere_index(3, 20402);

		set_sphere(sphere_pos, sphere_index);

		sphere_shader.bind();
		sphere_shader.uploadIndices(sphere_index);
		sphere_shader.uploadAttrib("position", sphere_pos);

	}

	~SimulationCanvas() {
		hair_shader.free();
		sphere_shader.free();
	}

	void init_mvp() {
		using namespace Eigen;
		using namespace nanogui;

		Matrix3f rotmat_y;
		rotmat_y <<
			cos(horizontalAngle), 0.0, sin(horizontalAngle)
			, 0.0, 1.0, 0.0
			, -sin(horizontalAngle), 0.0, cos(horizontalAngle);

		Matrix3f rotmat_x;
		rotmat_x <<
			1.0, 0.0, 0.0,
			0.0, cos(verticalAngle), -sin(verticalAngle),
			0.0, sin(verticalAngle), cos(verticalAngle);
		Matrix3f rotmat = rotmat_x * rotmat_y;

		Vector3f position = Vector3f(18.0, 25.0, 30.0);
		Vector3f target = Vector3f(0.0, -20.0, 0.0);
		position = rotmat * (Vector3f(0.0, 0.0, currentRadius) - target) + target;
		Vector3f up = rotmat * Vector3f(0.0, 1.0, 0.0);

		ModelMatrix = translate(Vector3f(0, 0, 0)) * get_orientate(0, 0, 0) * translate(Vector3f(0, -1.7, 0));
		ViewMatrix = lookAt(position, target, up);
		ProjectionMatrix = get_perspective(26.5, 6.0f / 6.0f, 0.1f, 2000.0f);
	}

	Eigen::Matrix4f get_perspective(float fovy, float aspect, float zNear, float zFar) {
		Eigen::Matrix4f mat;
		float tanHalfFovy = tan(fovy / 2.0);
		mat(0, 0) = (1) / (aspect * tanHalfFovy);
		mat(1, 1) = (1) / (tanHalfFovy);
		mat(2, 2) = -(zFar + zNear) / (zFar - zNear);
		mat(2, 3) = -(1);
		mat(3, 2) = -((2) * zFar * zNear) / (zFar - zNear);

		return mat;
	}

	Eigen::Matrix4f get_orientate(float yaw, float pitch, float roll) {
		float tmp_ch = cos(yaw);
		float tmp_sh = sin(yaw);
		float tmp_cp = cos(pitch);
		float tmp_sp = sin(pitch);
		float tmp_cb = cos(roll);
		float tmp_sb = sin(roll);

		Eigen::Matrix4f mat;
		mat(0,0) = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
		mat(0,1) = tmp_sb * tmp_cp;
		mat(0,2) = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
		mat(0,3) = float(0);
		mat(1,0) = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
		mat(1,1) = tmp_cb * tmp_cp;
		mat(1,2) = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
		mat(1,3) = float(0);
		mat(2,0) = tmp_sh * tmp_cp;
		mat(2,1) = -tmp_sp;
		mat(2,2) = tmp_ch * tmp_cp;
		mat(2,3) = float(0);
		mat(3,0) = float(0);
		mat(3,1) = float(0);
		mat(3,2) = float(0);
		mat(3,3) = float(1);
		return mat;
	}
	
	void set_sphere(nanogui::MatrixXf &pos, nanogui::MatrixXu &indices) {
		float radius = hm->params_host.sphere_rad;
		int i, j;
		int lats = 100;
		int longs = 100;
		int indicator = 1;
		float pi = 3.141592;
		int pos_index = 0;
		int indice_index = 0;
		// angular distance between two latitudinal lines
		float deltaPhi = -pi / longs;
		// angular distance between two longitudinal lines
		float deltaTheta = 2.0f * pi / lats;
		float3 c;
		float3 n;
		float2 u;
		float3 v;
		for (i = 0; i <= lats; i++) {
			// azimuth angles of the two arcs defining the stack (longitudes)
			float theta = i * deltaTheta;
			float thetaPrime = theta + deltaTheta;

			// projection of the first theta in the horizontal plane
			float x0 = cos(theta);
			float y0 = sin(theta);

			// projection of the second theta in the horizontal plane
			float x1 = cos(thetaPrime);
			float y1 = sin(thetaPrime);

			for (j = 0; j <= longs; j++) {

				c = make_float3(0.4, 0.4, 0.4);

				// polar angle
				float phi = j * deltaPhi;

				// polar vector in a vertical plane 
				float xPolar = cos(phi);
				float yPolar = sin(phi);

				// vertex #2 (theta , phiPrime)
				n = make_float3(yPolar * x1, yPolar * y1, xPolar);
				v = n * radius;
				u = make_float2((float)i / lats, (float)j / longs);
				v = v + hm->params_host.sphere_pos;
				v = v * 1.75f;
				pos.col(pos_index++) << v.x, v.y, v.z;
				//indices.push_back(indicator);
				indicator++;

				n = make_float3(yPolar * x0, yPolar * y0, xPolar);
				v = n * radius;
				v = v + hm->params_host.sphere_pos;
				u = make_float2((float)i / lats, (float)j / longs);
				v = v * 1.75f;
				pos.col(pos_index++) << v.x, v.y, v.z;
				//indices.push_back(indicator);
				indicator++;

				indices.col(indice_index++) << indicator - 3, indicator, indicator - 1;
				indices.col(indice_index++) << indicator - 2, indicator , indicator - 3;
			}
		}
	}

	void set_sphere_index(nanogui::MatrixXu &mat, int stackCount, int sectorCount) {
		int k1, k2;
		for (int i = 0; i < stackCount; ++i)
		{
			k1 = i * (sectorCount + 1);     // beginning of current stack
			k2 = k1 + sectorCount + 1;      // beginning of next stack

			for (int j = 0; j < sectorCount; ++j, ++k1, ++k2)
			{
				// 2 triangles per sector excluding first and last stacks
				// k1 => k2 => k1+1
				if (i != 0)
				{
					mat.col(i*sectorCount + j) << k1, k2, k1 + 1;
				}

				// k1+1 => k2 => k2+1
				if (i != (stackCount - 1))
				{
					mat.col(i*sectorCount + j) << k1+1, k2, k2 + 1;
				}
			}
		}
	}

	void setIndex(nanogui::MatrixXu &mat) {
		for (int i = 0; i < hm->STRAND_SIZE; i++) {
			for (int j = 0; j < hm->MAX_SIZE; j++) {
				if(j < hm->MAX_SIZE - 1)mat.col(i*hm->MAX_SIZE + j) << i * hm->MAX_SIZE + j, i * hm->MAX_SIZE + j + 1;
				else mat.col(i*hm->MAX_SIZE + j) << i * hm->MAX_SIZE + j, i * hm->MAX_SIZE + j;
			}
		}
	}

	void setPosition(float3 *v, nanogui::MatrixXf &mat) {
		for (int i = 0; i < hm->TOTAL_SIZE; i++) {
			mat.col(i) << v[i].x * 1.75f , v[i].y * 1.75f , v[i].z * 1.75f;
		}
	}

	void setColor(float3 *v, nanogui::MatrixXf &mat) {
		for (int i = 0; i < hm->STRAND_SIZE; i++) {
			for (int j = 0; j < hm->MAX_SIZE; j++) {
				mat.col(i*hm->MAX_SIZE + j) << v[i*hm->MAX_SIZE + j].x, v[i*hm->MAX_SIZE + j].y, v[i*hm->MAX_SIZE + j].z;
			}
		}
	}

	void setRotation(nanogui::Vector3f vRotation) {
		mRotation = vRotation;
	}

	virtual void drawGL() override {
		using namespace nanogui;

		hair_shader.bind();
		if (is_simulation) {

			hm->simulation();
			hm->get_colors();
			MatrixXu indices(2, hm->TOTAL_SIZE);
			setIndex(indices);
			
			MatrixXf positions(3, hm->TOTAL_SIZE);
			if (hm->state == COHESION_TEST)	setPosition(hm->particle_host.wet_position, positions);
			else setPosition(hm->particle_host.n_position, positions);
		
			hm->get_colors();
			MatrixXf colors(3, hm->TOTAL_SIZE);
			setColor(hm->colors, colors);

			hair_shader.uploadIndices(indices);
			hair_shader.uploadAttrib("position", positions);
			hair_shader.uploadAttrib("color", colors);
		}
		Matrix4f MVP = MVP.setIdentity();
		MVP = MVP + translate(Vector3f(-1.75, -1.5, -2.0));
		hair_shader.setUniform("modelViewProj", MVP);

		glEnable(GL_DEPTH_TEST);
		hair_shader.drawIndexed(GL_LINES, 0, hm->TOTAL_SIZE);
		if (hm->state == COLLISION_TEST || hm->state == ADHESION_TEST) {

			MatrixXf sphere_pos(3, 20402);
			MatrixXu sphere_index(3, 20402);

			set_sphere(sphere_pos, sphere_index);

			sphere_shader.bind();

			sphere_shader.uploadIndices(sphere_index);
			sphere_shader.uploadAttrib("position", sphere_pos);

			sphere_shader.setUniform("modelViewProj", MVP);
			sphere_shader.drawIndexed(GL_TRIANGLES, 0, 20402);
		}
		glDisable(GL_DEPTH_TEST);
	}
};

class GLTexture {
public:
	using handleType = std::unique_ptr<uint8_t[], void(*)(void*)>;
	GLTexture() = default;
	GLTexture(const std::string& textureName)
		: mTextureName(textureName), mTextureId(0) {}

	GLTexture(const std::string& textureName, GLint textureId)
		: mTextureName(textureName), mTextureId(textureId) {}

	GLTexture(const GLTexture& other) = delete;
	GLTexture(GLTexture&& other) noexcept
		: mTextureName(std::move(other.mTextureName)),
		mTextureId(other.mTextureId) {
		other.mTextureId = 0;
	}
	GLTexture& operator=(const GLTexture& other) = delete;
	GLTexture& operator=(GLTexture&& other) noexcept {
		mTextureName = std::move(other.mTextureName);
		std::swap(mTextureId, other.mTextureId);
		return *this;
	}
	~GLTexture() noexcept {
		if (mTextureId)
			glDeleteTextures(1, &mTextureId);
	}

	GLuint texture() const { return mTextureId; }
	const std::string& textureName() const { return mTextureName; }

	/**
	*  Load a file in memory and create an OpenGL texture.
	*  Returns a handle type (an std::unique_ptr) to the loaded pixels.
	*/
	handleType load(const std::string& fileName) {
		if (mTextureId) {
			glDeleteTextures(1, &mTextureId);
			mTextureId = 0;
		}
		int force_channels = 0;
		int w, h, n;
		handleType textureData(stbi_load(fileName.c_str(), &w, &h, &n, force_channels), stbi_image_free);
		if (!textureData)
			throw std::invalid_argument("Could not load texture data from file " + fileName);
		glGenTextures(1, &mTextureId);
		glBindTexture(GL_TEXTURE_2D, mTextureId);
		GLint internalFormat;
		GLint format;
		switch (n) {
		case 1: internalFormat = GL_R8; format = GL_RED; break;
		case 2: internalFormat = GL_RG8; format = GL_RG; break;
		case 3: internalFormat = GL_RGB8; format = GL_RGB; break;
		case 4: internalFormat = GL_RGBA8; format = GL_RGBA; break;
		default: internalFormat = 0; format = 0; break;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, GL_UNSIGNED_BYTE, textureData.get());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		return textureData;
	}

private:
	std::string mTextureName;
	GLuint mTextureId;
};

class MainScene : public nanogui::Screen {\
public:
	MainScene() : nanogui::Screen(Eigen::Vector2i(1920, 1080), "MainScene", false, false) {
		int w = 1920;
		int h = 1080;
		using namespace nanogui;
		//Simuation window
		Window *simulation_window = new Window(this, "Simulation window");
		simulation_window->setPosition(Vector2i(0, 0));
		simulation_window->setLayout(new BoxLayout(Orientation::Vertical,
			Alignment::Middle, 0, 5));
		
		Widget *tool_btn_panel = new Widget(simulation_window);
		tool_btn_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));

		Button *play_btn = new ToolButton(tool_btn_panel, ENTYPO_ICON_CONTROLLER_PLAY);
		play_btn->setFixedSize(nanogui::Vector2i(30, 30));
		play_btn->setCallback([this]() {
			is_simulation = true;
		});

		Button *stop_btn = new ToolButton(tool_btn_panel, ENTYPO_ICON_CONTROLLER_STOP);
		stop_btn->setFixedSize(nanogui::Vector2i(30, 30));
		stop_btn->setCallback([this]() {
			is_simulation = false;
		});

		Button *reset_btn = tool_btn_panel->add<Button>("", ENTYPO_ICON_CCW);
		reset_btn->setFixedSize(nanogui::Vector2i(30, 30));
		reset_btn->setCallback([this]() {
			float3 tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
			char *dir = hm->hair_style;
			hm = new HairModel(dir, num_strands);
			hm->color = tmp;
		});


		simulation_canvas = new SimulationCanvas(simulation_window);
		simulation_canvas->setBackgroundColor({ 100, 100, 100, 255 });
		simulation_canvas->setSize({w - w / 3 - 50, h - 50});

		//Hair style window
		Window *hair_style_window = new Window(this, "Select hair style");
		hair_style_window->setPosition(Vector2i(w - w / 3, 0));
		vector<pair<int, string>>icons = loadImageDirectory(mNVGContext, "icons");
		new Label(hair_style_window, "Image panel & scroll panel", "sans-bold");
		PopupButton *imagePanelBtn = new PopupButton(hair_style_window, "Select style");
		imagePanelBtn->setIcon(ENTYPO_ICON_FOLDER);
		Popup *popup = imagePanelBtn->popup();
		VScrollPanel *vscroll = new VScrollPanel(popup);
		ImagePanel *imgPanel = new ImagePanel(vscroll);
		imgPanel->setImages(icons);
		popup->setFixedSize(Vector2i(245, h / 3 - 100));

		imgPanel->setCallback([this](int i) {
			float3 tmp;
			switch (i)
			{
			case 0:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00000.txt", num_strands);
				hm->color = tmp;
				break;
			case 1:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00001.txt", num_strands);
				hm->color = tmp;
				break;
			case 2:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00002.txt", num_strands);
				hm->color = tmp;
				break;
			case 3:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00003.txt", num_strands);
				hm->color = tmp;
				break;
			case 4:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00004.txt", num_strands);
				hm->color = tmp;
				break;
			}
		});

		GridLayout *layout =
			new GridLayout(Orientation::Horizontal, 2,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
			{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		hair_style_window->setLayout(layout);

		new Label(hair_style_window, "Color picker :", "sans-bold");
		auto cp = new ColorPicker(hair_style_window, { 255, 200, 0, 255 });
		cp->setFixedSize({ 150, 30 });
		cp->setFinalCallback([](const Color &c) {
			hm->color = make_float3(c.r(), c.g(), c.b());
			std::cout << "Set color : ["
				<< c.r() << ", "
				<< c.g() << ", "
				<< c.b() << ", "
				<< c.w() << "]" << std::endl;
		});

		new Label(hair_style_window, "Hair density", "sans-bold");
		Widget *hair_density_panel = new Widget(hair_style_window);
		hair_density_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		Slider *hair_density_slider = new Slider(hair_density_panel);

		hair_density_slider->setRange(std::pair<float, float>(64, 512));
		hair_density_slider->setValue(128);
		hair_density_slider->setFixedWidth(160);
		TextBox *hair_density_textBox = new TextBox(hair_density_panel);
		hair_density_textBox->setFixedSize(Vector2i(64, 25));
		hair_density_textBox->setValue("128");

		hair_density_slider->setCallback([hair_density_textBox](float value) {
			hair_density_textBox->setValue(std::to_string((int)(value)));
			float3 tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
			char *dir = hm->hair_style;
			num_strands = (int)value;
			hm = new HairModel(dir, num_strands);
			hm->color = tmp;
		});


		//Physics test window
		Window *physics_window = new Window(this, "Physics test window");
		physics_window->setPosition(Vector2i(w - w / 3, h / 3));

		layout =
			new GridLayout(Orientation::Horizontal, 2,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
			{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		physics_window->setLayout(layout);

		new Label(physics_window, "General", "sans-bold");
		Button *rb = new Button(physics_window, "General Simulation");
		rb->setFlags(Button::RadioButton);
		rb->setCallback([]() {
			hm->state = GENERAL_SIMULATION;
		});
		

		new Label(physics_window, "Bouncing test", "sans-bold");
		rb = new Button(physics_window, "Bouncing Simulation");
		rb->setFlags(Button::RadioButton);
		rb->setCallback([]() {
			hm->state = BOUNCING_TEST;
		});

		new Label(physics_window, "Rotate test", "sans-bold");
		rb = new Button(physics_window, "Rotate Simulation");
		rb->setFlags(Button::RadioButton);
		rb->setCallback([]() {
			hm->state = ROTATE_TEST;
		});

		new Label(physics_window, "Collision test", "sans-bold");
		rb = new Button(physics_window, "Collision Simulation");
		rb->setFlags(Button::RadioButton);
		rb->setCallback([]() {
			hm->state = COLLISION_TEST;
		});

		new Label(physics_window, "Cohesion test", "sans-bold");
		rb = new Button(physics_window, "Cohesion Simulation");
		rb->setFlags(Button::RadioButton);
		rb->setCallback([]() {
			hm->state = COHESION_TEST;
		});

		new Label(physics_window, "Adhesion test", "sans-bold");
		rb = new Button(physics_window, "Adhesion Simulation");
		rb->setFlags(Button::RadioButton);
		rb->setCallback([]() {
			hm->state = ADHESION_TEST;
		});

		Window *parameter_window = new Window(this, "Parmameter setting");
		parameter_window->setPosition(Vector2i(w - w / 3, h / 3 * 2));

		layout =
			new GridLayout(Orientation::Horizontal, 2,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
			{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		parameter_window->setLayout(layout);

		new Label(parameter_window, "Stretch spring coefficient", "sans-bold");
		Widget *parameter_panel = new Widget(parameter_window);
		parameter_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		Slider *stretch_slider = new Slider(parameter_panel);

		stretch_slider->setRange(pair<float, float>(0, 500000 * 2));
		stretch_slider->setValue(500000);
		stretch_slider->setFixedWidth(160);
		TextBox *stretch_textBox = new TextBox(parameter_panel);
		stretch_textBox->setFixedSize(Vector2i(100, 25));
		stretch_textBox->setValue("500000");

		stretch_slider->setCallback([stretch_textBox](float value) {
			stretch_textBox->setValue(std::to_string((int)(value)));
			hm->params_host.K_S = value;
			hm->set_parameter();
		});

		new Label(parameter_window, "Bending spring coefficient", "sans-bold");
		parameter_panel = new Widget(parameter_window);
		parameter_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));

		Slider *bending_slider = new Slider(parameter_panel);
		bending_slider->setRange(pair<float, float>(0, 60000));
		bending_slider->setValue(30000);
		bending_slider->setFixedWidth(160);
		TextBox *bending_textBox = new TextBox(parameter_panel);
		bending_textBox->setFixedSize(Vector2i(100, 25));
		bending_textBox->setValue("30000");

		bending_slider->setCallback([bending_textBox](float value) {
			bending_textBox->setValue(std::to_string((int)(value)));
			hm->params_host.K_B = value;
			hm->set_parameter();
		});

		new Label(parameter_window, "Core spring coefficient", "sans-bold");
		parameter_panel = new Widget(parameter_window);
		parameter_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		Slider *core_slider = new Slider(parameter_panel);

		core_slider->setRange(std::pair<float, float>(0, 30000));
		core_slider->setValue(15000);
		core_slider->setFixedWidth(160);
		TextBox *core_textBox = new TextBox(parameter_panel);
		core_textBox->setFixedSize(Vector2i(100, 25));
		core_textBox->setValue("15000");

		core_slider->setCallback([core_textBox](float value) {
			core_textBox->setValue(std::to_string((int)(value)));
			hm->params_host.K_C = value;
			hm->set_parameter();
		});

		new Label(parameter_window, "Saturation coefficient", "sans-bold");
		parameter_panel = new Widget(parameter_window);
		parameter_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		Slider *saturation_slider = new Slider(parameter_panel);

		saturation_slider->setRange(std::pair<float, float>(0, 30000));
		saturation_slider->setValue(0);
		saturation_slider->setFixedWidth(160);
		TextBox *saturation_textBox = new TextBox(parameter_panel);
		saturation_textBox->setFixedSize(Vector2i(100, 25));
		saturation_textBox->setValue("0");

		saturation_slider->setCallback([saturation_textBox](float value) {
			saturation_textBox->setValue(std::to_string((int)(value)));
			hm->params_host.R_C = value;
			hm->set_parameter();
		});

		Widget *reset_panel = new Widget(parameter_window);
		reset_panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		reset_btn = reset_panel->add<Button>("", ENTYPO_ICON_CCW);
		reset_btn->setFixedSize(nanogui::Vector2i(30, 30));
		reset_btn->setCallback([stretch_slider, stretch_textBox, bending_slider, bending_textBox,
		core_slider, core_textBox, saturation_slider, saturation_textBox]() {
			hm->params_host.K_S = 500000;
			hm->params_host.K_B = 30000;
			hm->params_host.K_C = 15000;
			hm->params_host.R_C = 0; 
			hm->set_parameter();

			stretch_slider->setValue(500000);
			bending_slider->setValue(30000);
			core_slider->setValue(15000);
			saturation_slider->setValue(0);

			stretch_textBox->setValue("500000");
			bending_textBox->setValue("30000");
			core_textBox->setValue("15000");
			saturation_textBox->setValue("0");
		});
		new Label(reset_panel, "                                   ", "sans-bold");

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
	using imagesDataType = vector<pair<GLTexture, GLTexture::handleType>>;
	imagesDataType mImagesData;
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
		std::cerr << error_msg << endl;
		return -1;
	}

	return 0;
}