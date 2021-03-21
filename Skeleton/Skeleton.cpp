//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : jdp18v
// Neptun : Kovari Daniel Mate
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		float w = sqrt(vp.x*vp.x + vp.y*vp.y + 1);
		gl_Position = vec4(vp.x/w, vp.y/w, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char *const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
const int GRAPHPOINTS = 50;	// numof graph points
float verticesCoordinates[GRAPHPOINTS * 2] = { 0 };		// array of coordinates
float mouseXPrev;
float mouseYPrev;
float mouseXNext;
float mouseYNext;
float vertices[GRAPHPOINTS * 3 * 2 * 2];		// because we're building squares, which is two triangles  (triangles*2 points for coordinates*two triangles)
unsigned int vbo;		// vertex buffer object

vec3 mouseMoves(float pX, float pY, float coordX, float coordY);
vec3 getDivider(vec3 a, vec3 b, float ratio);
vec3 getMirrorOnPoint(vec3 p, vec3 on);
void printVec3(vec3 in);
float calcW(float x, float y);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	const float SIDELENGTH = 0.1;
	//float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };

	int j = 0;
	for (int i = 0; i < GRAPHPOINTS * 3 * 2 * 2; i += 12)		//generating graphpoints
	{
		float randX = (rand() % 100 - 50) / 10.0;		// random num between 
		float randY = (rand() % 100 - 50) / 10.0;
		verticesCoordinates[j] = randX;
		verticesCoordinates[j + 1] = randY;
		j += 2;
		vertices[i + 0] = randX - SIDELENGTH / 2;			// first corner of square
		vertices[i + 1] = randY - SIDELENGTH / 2;
		vertices[i + 2] = randX - SIDELENGTH / 2;
		vertices[i + 3] = randY + SIDELENGTH / 2;
		vertices[i + 4] = randX + SIDELENGTH / 2;
		vertices[i + 5] = randY - SIDELENGTH / 2;


		vertices[i + 6] = randX - SIDELENGTH / 2;		// second triangle coordinates
		vertices[i + 7] = randY + SIDELENGTH / 2;
		vertices[i + 8] = randX + SIDELENGTH / 2;
		vertices[i + 9] = randY - SIDELENGTH / 2;
		vertices[i + 10] = randX + SIDELENGTH / 2;
		vertices[i + 11] = randY + SIDELENGTH / 2;
	}

	for (int i = 0; i < GRAPHPOINTS * 2; i++)
	{
		printf("X:%3.2f Y:%3.2f\n", verticesCoordinates[i], verticesCoordinates[i + 1]);
	}


	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	//	in case of motion
	for (int i = 0; i < GRAPHPOINTS * 3 * 2 * 2; i += 2)		//generating graphpoints
	{
		//vec3 a = {0.1f, 0.0f, 1.0f};
		//vec3 b = (mouseX, mouseY, (mouseX * mouseX + mouseY * mouseY + 1));
		//vec3 b = { 0.0f, 0.0f, 1.0f };
		//vec3 m1 = getDivider(a, b, 0.25f);
		//printf("\n\nm1\n");
		//printVec3(m1);
		//printf("\n");
		//vec3 m2 = getDivider(a, b, 0.75f);
		//printVec3(m2);

		//vec3 m1 = { 0.0f, 0.0f, 1.0f };
		//vec3 m2 = { 0.0f, 0.0f, 1.0f };

		vec3 m1 = { mouseXPrev, mouseYPrev, calcW(mouseXPrev, mouseYPrev) };
		vec3 m2 = { mouseXNext, mouseYNext, calcW(mouseXNext, mouseYNext) };
		vec3 temp = { vertices[i + 0], vertices[i + 1], calcW(vertices[i + 0], vertices[i + 1]) };
		printVec3(temp);
		temp = getMirrorOnPoint(temp, m1);
		temp = getMirrorOnPoint(temp, m2);
		//printVec3(temp);
		vertices[i + 0] = temp.x;			// X coordinate of said point
		vertices[i + 1] = temp.y;			// Y coordinate of said point
	}


	//
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location


	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 * GRAPHPOINTS * 2 /*# Elements*/);		// 3 for 3 points * graph points * 2 triangles each

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	mouseXPrev = mouseXNext;
	mouseYPrev = mouseYNext;
	mouseXNext = cX;
	mouseYNext = cY;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char *buttonStat;
	switch (state) {
		case GLUT_DOWN: buttonStat = "pressed"; break;
		case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
		case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
		case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
		case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}


// Gets distance between 2 vec3 vectors
float vec3Distance(vec3 p, vec3 q) {
	printf("TAVOLSAG\n");
	printVec3(q);
	printVec3(p);
	printf("TAVOLSAGvege\n");
	printf("%3.5f", -(p.x * q.x + p.y * q.y - p.z * q.z));
	return acosh(-(p.x * q.x + p.y * q.y - p.z * q.z));
}

// Gets a point from a vec3, a direction vetor and a distance
vec3 pointFromVDir(vec3 p, vec3 v, float t) {
	return (p * cosh(t) + v * sinh(t));
}

// Gets a direction vector from 2 points and a distance
vec3 dirVecFrom2Points(vec3 p, vec3 q, float d_pq) {
	return ((q - p * cosh(d_pq)) / (sinh(d_pq)));
}

// Gets Mn from a->b line equation
vec3 getDivider(vec3 a, vec3 b, float ratio) {
	float d_ab = vec3Distance(a, b);
	printf("d_ab = %3.5f\n", d_ab);
	vec3 v = dirVecFrom2Points(a, b, d_ab);
	return (pointFromVDir(a, v, ratio * d_ab));
}

// Mirrors one time
vec3 getMirrorOnPoint(vec3 p, vec3 on) {
	return getDivider(p, on, 2);
}


//// Moves the graph
//vec3 mouseMoves(float pX, float pY, float coordX, float coordY) {
//	//1,
//	vec3 q = vec3(pX, pY, sqrtf(pX * pX + pY * pY + 1.0f));
//	vec3 p = vec3(0.0f, 0.0f, 1.0f);
//	vec3 v = vec3(coordX, coordY, sqrtf(coordX * coordX + coordY * coordY + 1.0f));
//	float d_pq = vec3Distance(p, q);
//	//2,
//	vec3 v_dir1 = dirVecFrom2Points(p, q, d_pq);
//	//3,
//	vec3 m = pointFromVDir(p, v, d_pq / 2.0f);
//	//4,
//	float d_vm = vec3Distance(v, m);
//	//5,
//	vec3 v_dir2 = dirVecFrom2Points(v, m, d_vm);
//	//6,
//	vec3 v1 = pointFromVDir(v, v_dir2, 2 * d_vm);
//	//7,
//	float d_v1q = vec3Distance(v1, q);
//	//8,
//	vec3 v2 = pointFromVDir(v1, q, d_v1q * 2.0f);		// the point where the graph point should go
//
//	return v2;
//}

void printVec3(vec3 in) {
	printf("X: %3.2f Y:%3.2f W:%3.2f\n", in.x, in.y, in.z);
}

// calculates w from given X and Y coordinates
float calcW(float x, float y) {
	return (sqrtf(x * x + y * y + 1));
}