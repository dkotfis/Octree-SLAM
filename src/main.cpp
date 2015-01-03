
#include "main.h"

string path_prefix = "../../";

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

#if defined(__GNUC__)
  path_prefix = "";
#endif

	bool loadedScene = false;

	int choice = 2;
	cout << "Please enter which scene to load? '1'(dragon), '2'(cow), '3'(bunny)." << endl;
	cin >> choice;

	string local_path = path_prefix + "../objs/";
	string data = local_path+ "2cows.obj";
	if (choice == 1)
		data = local_path+ "dragon_tex.obj";
	else if (choice == 2)
		data = local_path + "cow_tex.obj";
  else if (choice == 3)
    data = local_path + "bunny_tex.obj";

	mesh = new obj();
	objLoader* loader = new objLoader(data, mesh);
	mesh->buildVBOs();
	meshes.push_back(mesh);
	delete loader;
	loadedScene = true;


	frame = 0;
	seconds = time (NULL);
	fpstracker = 0;

	// Launch CUDA/GL
	if (init(argc, argv)) {

		//Voxelize the scene
		if (VOXELIZE) {
		  voxelizeScene();
		}

		// GLFW main loop
		mainLoop();
	}

	system("PAUSE");
	return 0;
}


void mainLoop() {
	while(!glfwWindowShouldClose(window)){
		glfwPollEvents();

    computeMatricesFromInputs();

		if (USE_CUDA_RASTERIZER) {
			runCuda();
		} else {
			runGL();
		}

		time_t seconds2 = time (NULL);

		if(seconds2-seconds >= 1){

			fps = fpstracker/(seconds2-seconds);
			fpstracker = 0;
			seconds = seconds2;
		}

		string title = "Voxel Rendering | " + utilityCore::convertIntToString((int)fps) + " FPS";
		glfwSetWindowTitle(window, title.c_str());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if (USE_CUDA_RASTERIZER) {
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(GL_TEXTURE_2D, displayImage);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);  
			glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
		} else {
			glDrawArrays(GL_TRIANGLES, 0, vbosize);
		}

		glfwSwapBuffers(window);
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void computeMatricesFromInputs() {

  //Compute the current time
  double currentTime = glfwGetTime();
  float deltaTime = float(currentTime = lastTime);
  lastTime = currentTime;

  //Only use the mouse if the left button is clicked
  if (LB) {
    //Read the mouse position
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos); 

    //Reset the mouse position to the center
    glfwSetCursorPos(window, width/2, height/2);

    //Compute the viewing angle
    horizontalAngle += mouseSpeed * deltaTime * float(width/2 - xpos);
    verticalAngle += mouseSpeed * deltaTime * float(height/2 - ypos);
  }

  //Compute the direction
  glm::vec3 direction(cos(verticalAngle) * sin(horizontalAngle), sin(verticalAngle), cos(verticalAngle)*cos(horizontalAngle));
  glm::vec3 right = glm::vec3(sin(horizontalAngle - 3.14f/2.0f), 0, cos(horizontalAngle - 3.14f/2.0f));
  glm::vec3 up = glm::cross(right, direction);

  //Compute position from keyboard inputs
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
    position += direction * deltaTime * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
    position -= direction * deltaTime * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
    position += right * deltaTime * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
    position -= right * deltaTime * speed;
  }

  //Update the matrices
  model = glm::mat4(1.0f);
  view = glm::lookAt(position, position+direction, up);
  projection = glm::perspective(FoV, (float)(width) / (float)(height), zNear, zFar);
  modelview = view * model;
  mvp = projection*modelview;

}

void runCuda() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr=NULL;

	glm::mat4 rotationM = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(1.0f, 0.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 20.0f-0.5f*frame, glm::vec3(0.0f, 1.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));

  float newcbo[] = { 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0 };

	//Update data
	if (VOXELIZE) {
		vbo = m_vox.vbo;
		vbosize = m_vox.vbosize;
    cbo = newcbo;
    cbosize = 9;
		ibo = m_vox.ibo;
		ibosize = m_vox.ibosize;
		nbo = m_vox.nbo;
		nbosize = m_vox.nbosize;
	} else {
		vbo = mesh->getVBO();
		vbosize = mesh->getVBOsize();
		cbo = newcbo;
		cbosize = 9;
		ibo = mesh->getIBO();
		ibosize = mesh->getIBOsize();
		nbo = mesh->getNBO();
		nbosize = mesh->getNBOsize();
	}

	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(width, height), rotationM, frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, &tex, texcoord, view, lightpos, mode, barycenter);
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;

	frame++;
	fpstracker++;

}

void runGL() {

  float newcbo[] = { 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0 };

	//Update data
	if (VOXELIZE) {
		vbo = m_vox.vbo;
		vbosize = m_vox.vbosize;
    cbo = m_vox.cbo;
    cbosize = m_vox.cbosize;
		ibo = m_vox.ibo;
		ibosize = m_vox.ibosize;
		nbo = m_vox.nbo;
		nbosize = m_vox.nbosize;
	} else {
		vbo = mesh->getVBO();
		vbosize = mesh->getVBOsize();
		cbo = newcbo;
		cbosize = 9;
		ibo = mesh->getIBO();
		ibosize = mesh->getIBOsize();
		nbo = mesh->getNBO();
		nbosize = mesh->getNBOsize();
	}

	//Send the MV, MVP, and Normal Matrices
	glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm::value_ptr(mvp));
	glUniformMatrix4fv(proj_location, 1, GL_FALSE, glm::value_ptr(projection));
	glm::mat3 norm_mat = glm::mat3(glm::transpose(glm::inverse(model)));
	glUniformMatrix3fv(norm_location, 1, GL_FALSE, glm::value_ptr(norm_mat));

	//Send the light position
	glUniform3fv(light_location, 1, glm::value_ptr(lightpos));

	// Send the VBO and NB0
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, vbosize*sizeof(float), vbo, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, nbosize*sizeof(float), nbo, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);

  if (VOXELIZE) {
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ARRAY_BUFFER, cbosize*sizeof(float), cbo, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(2);
  }

	frame++;
	fpstracker++;
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void voxelizeScene() {

	//Construct target mesh
	Mesh m_in;
	m_in.vbo = mesh->getVBO();
	m_in.vbosize = mesh->getVBOsize();
	m_in.nbo = mesh->getNBO();
	m_in.nbosize = mesh->getNBOsize();
	m_in.cbo = mesh->getCBO();
	m_in.cbosize = mesh->getCBOsize();
	m_in.ibo = mesh->getIBO();
	m_in.ibosize = mesh->getIBOsize();
  m_in.tbo = mesh->getTBO();
  m_in.tbosize = mesh->getTBOsize();

	//Load cube
	Mesh m_cube;
	obj* cube = new obj();
	string cubeFile = path_prefix + "../objs/cube.obj";
	objLoader* loader = new objLoader(cubeFile, cube);
	cube->buildVBOs();
	m_cube.vbo = cube->getVBO();
	m_cube.vbosize = cube->getVBOsize();
	m_cube.nbo = cube->getNBO();
	m_cube.nbosize = cube->getNBOsize();
	m_cube.cbo = cube->getCBO();
	m_cube.cbosize = cube->getCBOsize();
	m_cube.ibo = cube->getIBO();
	m_cube.ibosize = cube->getIBOsize();
	delete cube;

	//Voxelize
  if (OCTREE){
    voxelizeSVOCubes(m_in, &tex, m_cube, m_vox);
  } else {
    voxelizeToCubes(m_in, &tex, m_cube, m_vox);
  }
}

//read .bmp texture and assign it to tex
void readBMP(const char* filename, bmp_texture &tex)
{
	int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);
	  glm::vec3 *color_data = new glm::vec3[size/3];
    for(i = 0; i < size; i += 3){
			color_data[i/3].r = (int)data[i+2]/255.0f;
			color_data[i/3].g = (int)data[i+1]/255.0f;
			color_data[i/3].b = (int)data[i]/255.0f;
    }
    delete []data;
	tex.data = color_data;
	tex.height = height;
	tex.width = width;
}


bool init(int argc, char* argv[]) {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		return false;
	}

	readBMP((path_prefix + string("../textures/texture1.bmp")).c_str(), tex);
	width = 800;
	height = 800;
	window = glfwCreateWindow(width, height, "Voxel Rendering", NULL, NULL);
	if (!window){
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetMouseButtonCallback(window, MouseClickCallback);
	glfwSetScrollCallback(window, ScrollCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if(glewInit()!=GLEW_OK){
		return false;
	}

	// Initialize other stuff
	if (USE_CUDA_RASTERIZER) {
		initCudaTextures();
		initCudaVAO();
		initCuda();
		initCudaPBO();
		initPassthroughShaders();
		glActiveTexture(GL_TEXTURE0);
	} else {
		initGL();
		initDefaultShaders();
	}

  // Initialize the time
  lastTime = glfwGetTime();

	return true;
}

void initCudaPBO(){
	// set up vertex data parameter
	int num_texels = width*height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void initCuda(){
	// Use device with highest Gflops/s
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initCudaTextures(){
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

void initCudaVAO(void){
	GLfloat vertices[] =
	{ 
		-1.0f, -1.0f, 
		1.0f, -1.0f, 
		1.0f,  1.0f, 
		-1.0f,  1.0f, 
	};

	GLfloat texcoords[] = 
	{ 
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initPassthroughShaders() {
	const char *attribLocations[] = { "Position", "Tex" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}

void initGL() {

	glGenBuffers(3, buffers);
	glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

}

GLuint initDefaultShaders() {
	const char *attribLocations[] = { "v_position", "v_normal" };

  string vs, fs;
  if (VOXELIZE) {
    vs = path_prefix + "../shaders/voxels.vert";
    fs = path_prefix + "../shaders/voxels.frag";
  } else {
    vs = path_prefix + "../shaders/default.vert";
    fs = path_prefix + "../shaders/default.frag";
  }
	const char *vertShader = vs.c_str();
  const char *fragShader = fs.c_str();

	GLuint program = glslUtility::createProgram(attribLocations, 2, vertShader, fragShader);

	glUseProgram(program);
	mvp_location = glGetUniformLocation(program, "u_mvpMatrix");
	proj_location = glGetUniformLocation(program, "u_projMatrix");
	norm_location = glGetUniformLocation(program, "u_normMatrix");
	light_location = glGetUniformLocation(program, "u_light");

	return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
	if(pbo) deletePBO(&pbo);
	if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex){
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void shut_down(int return_code){
	kernelCleanup();
	cudaDeviceReset();
#ifdef __APPLE__
	glfwTerminate();
#endif
	exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char* description){
	fputs(description, stderr);
}

void MouseClickCallback(GLFWwindow *window, int button, int action, int mods){
  if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
		LB = true;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  }

  if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT) {
		LB = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  }
}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  //Update the field of view from the mouse scroll wheel
  FoV -= 2 * yoffset;
}
