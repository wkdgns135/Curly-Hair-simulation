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
bool is_simulation = true;

class SimulationCanvas : public nanogui::GLCanvas {
public:
	SimulationCanvas(Widget *parent) : nanogui::GLCanvas(parent), mRotation(nanogui::Vector3f(0.25f, 0.5f, 0.33f)) {
		using namespace nanogui;
		hm = new HairModel("strands\\strands00000.txt");
		hm->color = make_float3(1.0, 0.8, 0.0);

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

		if (is_simulation) {
			hm->simulation();
			hm->get_colors();


			MatrixXu indices(2, hm->TOTAL_SIZE); /* Draw a cube */
			setIndex(indices);

			MatrixXf positions(3, hm->TOTAL_SIZE);
			setPosition(hm->particle_host.position, positions);

			hm->get_colors();
			MatrixXf colors(3, hm->TOTAL_SIZE);
			setColor(hm->colors, colors);

			mShader.uploadIndices(indices);
			mShader.uploadAttrib("position", positions);
			mShader.uploadAttrib("color", colors);
		}
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
	MainScene() : nanogui::Screen(Eigen::Vector2i(1980, 1080), "MainScene", false, true) {
		using namespace nanogui;
		//Simuation window
		Window *window = new Window(this, "Simulation window");
		window->setPosition(Vector2i(0, 0));
		window->setLayout(new BoxLayout(Orientation::Vertical,
			Alignment::Middle, 0, 5));
		
		Widget *panel = new Widget(window);
		panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));

		ToolButton *b = new ToolButton(panel, ENTYPO_ICON_CONTROLLER_PLAY);
		b->setCallback([this]() {
			is_simulation = true;
		});

		b = new ToolButton(panel, ENTYPO_ICON_CONTROLLER_STOP);
		b->setCallback([this]() {
			is_simulation = false;
		});
		b = new ToolButton(panel, ENTYPO_ICON_CCW);
		b->setCallback([this]() {
			float3 tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
			char *dir = hm->hair_style;
			hm = new HairModel(dir);
			hm->color = tmp;
		});


		simulation_canvas = new SimulationCanvas(window);
		simulation_canvas->setBackgroundColor({ 100, 100, 100, 255 });
		simulation_canvas->setSize({1980 - 1980 / 3 - 50, 1080 - 50});

		//Hair style window
		window = new Window(this, "Select hair style");
		window->setPosition(Vector2i(1980 - 1980 / 3, 0));
		vector<pair<int, string>>icons = loadImageDirectory(mNVGContext, "icons");
		new Label(window, "Image panel & scroll panel", "sans-bold");
		PopupButton *imagePanelBtn = new PopupButton(window, "Select style");
		imagePanelBtn->setIcon(ENTYPO_ICON_FOLDER);
		Popup *popup = imagePanelBtn->popup();
		VScrollPanel *vscroll = new VScrollPanel(popup);
		ImagePanel *imgPanel = new ImagePanel(vscroll);
		imgPanel->setImages(icons);
		popup->setFixedSize(Vector2i(245, 1080 / 3 - 100));

		imgPanel->setCallback([this](int i) {
			float3 tmp;
			switch (i)
			{
			case 0:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00000.txt");
				hm->color = tmp;
				break;
			case 1:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00001.txt");
				hm->color = tmp;
				break;
			case 2:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00002.txt");
				hm->color = tmp;
				break;
			case 3:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00003.txt");
				hm->color = tmp;
				break;
			case 4:
				tmp = make_float3(hm->color.x, hm->color.y, hm->color.z);
				hm = new HairModel("strands\\strands00004.txt");
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
		window->setLayout(layout);

		new Label(window, "Color picker :", "sans-bold");
		auto cp = new ColorPicker(window, { 255, 200, 0, 255 });
		cp->setFixedSize({ 150, 30 });
		cp->setFinalCallback([](const Color &c) {
			hm->color = make_float3(c.r(), c.g(), c.b());
			std::cout << "Set color : ["
				<< c.r() << ", "
				<< c.g() << ", "
				<< c.b() << ", "
				<< c.w() << "]" << std::endl;
		});

		//Physics test window
		window = new Window(this, "Physics test window");
		window->setPosition(Vector2i(1980 - 1980 / 3, 1080 / 3));




		window = new Window(this, "Parmameter setting");
		window->setPosition(Vector2i(1980 - 1980 / 3, 1080 / 3 * 2));

		layout =
			new GridLayout(Orientation::Horizontal, 2,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
			{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		window->setLayout(layout);

		new Label(window, "Stretch spring coefficient", "sans-bold");
		panel = new Widget(window);
		panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		Slider *slider = new Slider(panel);

		slider->setRange(pair<float, float>(0, 500000 * 2));
		slider->setValue(500000);
		slider->setFixedWidth(160);
		TextBox *textBox = new TextBox(panel);
		textBox->setFixedSize(Vector2i(100, 25));
		textBox->setValue("500000");

		slider->setCallback([textBox](float value) {
			textBox->setValue(std::to_string((int)(value)));
			hm->params_host.K_S = value;
			hm->set_parameter();
		});

		new Label(window, "Bending spring coefficient", "sans-bold");
		panel = new Widget(window);
		panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		slider = new Slider(panel);

		slider->setRange(pair<float, float>(0, 60000));
		slider->setValue(30000);
		slider->setFixedWidth(160);
		textBox = new TextBox(panel);
		textBox->setFixedSize(Vector2i(100, 25));
		textBox->setValue("30000");

		slider->setCallback([textBox](float value) {
			textBox->setValue(std::to_string((int)(value)));
			hm->params_host.K_B = value;
			hm->set_parameter();
		});

		new Label(window, "Core spring coefficient", "sans-bold");
		panel = new Widget(window);
		panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		slider = new Slider(panel);

		slider->setRange(std::pair<float, float>(0, 30000));
		slider->setValue(15000);
		slider->setFixedWidth(160);
		textBox = new TextBox(panel);
		textBox->setFixedSize(Vector2i(100, 25));
		textBox->setValue("15000");

		slider->setCallback([textBox](float value) {
			textBox->setValue(std::to_string((int)(value)));
			hm->params_host.K_C = value;
			hm->set_parameter();
		});

		new Label(window, "Saturation coefficient", "sans-bold");
		panel = new Widget(window);
		panel->setLayout(new BoxLayout(Orientation::Horizontal,
			Alignment::Middle, 0, 5));
		slider = new Slider(panel);

		slider->setRange(std::pair<float, float>(0, 30000));
		slider->setValue(0);
		slider->setFixedWidth(160);
		textBox = new TextBox(panel);
		textBox->setFixedSize(Vector2i(100, 25));
		textBox->setValue("0");

		slider->setCallback([textBox](float value) {
			textBox->setValue(std::to_string((int)(value)));
			hm->params_host.R_C = value;
			hm->set_parameter();
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