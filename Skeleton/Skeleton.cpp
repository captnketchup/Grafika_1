//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
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
	layout(location = 1) in vec2 UV;
	layout(location = 2) in float seed;

	out vec2 texCoord;
	out float funcSeed;

	void main() {
		float w = sqrt(vp.x*vp.x + vp.y*vp.y + 1);
		gl_Position = vec4(vp.x/w, vp.y/w, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
		texCoord = UV;
		funcSeed = seed;
	}
)";


// fragment shader in GLSL
const char *const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
	in vec2 texCoord;		
	in float funcSeed;

	void main() {
		//outColor = vec4(color, 1);	// computed color is the color of the primitive
		if(texCoord.x * texCoord.x + texCoord.y * texCoord.y < 1){
			outColor = vec4(sin(funcSeed)/2+1, sin(funcSeed+1)/2+1, sin(funcSeed+2)/2+1, 1);	// computed color is the color of the primitive
		}
		else{
			outColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}
)";



GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
unsigned int vaoLines;		// vao for the edges between graphpoints
const int GRAPHPOINTS = 50;	// numof graph points
float verticesCoordinates[GRAPHPOINTS * 2] = { 0 };		// array of coordinates
vec2 velocity[50] = { 0.0f, 0.0f };
float mouseXPrev = 0.0f;
float mouseYPrev = 0.0f;
float mouseXNext = 0.0f;
float mouseYNext = 0.0f;
float vertices[GRAPHPOINTS * 3 * 2 * 2];		// because we're building squares, which is two triangles  (triangles*2 points for coordinates*two triangles)
unsigned int vbo[4];		// vertex buffer object	 0: vertices(négyzetcsúcsok)	1: edges(élek)		2: textúra		3: színek
//unsigned int vboLines;		// vbo for the edges between graphpoints
const int NUMOFEDGES = (int)(GRAPHPOINTS * (GRAPHPOINTS - 1.0f) / 2.0f * 0.05f);
//vec2 graphEdges[NUMOFEDGES * 2] = { 0.0f, 0.0f };		// relation between points which one has an edge
bool adjacencyMtx[GRAPHPOINTS][GRAPHPOINTS] = { false };
float verticesLines[NUMOFEDGES * 4];		// number of edges between graphpoints
float textureCoordinates[GRAPHPOINTS * 3 * 2 * 2];
float colorSeeds[GRAPHPOINTS * 3 * 2];
const float FRICTION = 0.7f;
vec2 centreOfMass = { 0.0f, 0.0f };
bool isSpace = false;
const float optimalDistance = 2.0f;
const float SIDELENGTH = 0.05;


//vec3 mouseMoves(float pX, float pY, float coordX, float coordY);
vec3 getDivider(vec3 a, vec3 b, float ratio);
vec3 getMirrorOnPoint(vec3 p, vec3 on);
void printVec3(vec3 in);
float calcW(float x, float y);
vec3 calcHyperbolicCoord(float x, float y);
vec3 calcVec3withW(float x, float y);
vec3 pointFromVDir(vec3 p, vec3 v, float t);
vec2 sumForce(int idxX);
void graphMoves();
vec2 calcCentreOfMass(vec2 in);
vec2 calcMovingVector(float x, float y, vec3 m1, vec3 m2);
float vec3Distance(vec3 p, vec3 q);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);		//lehet nem lehet használni xdd
	glEnable(GL_BLEND);

	glGenVertexArrays(1, &vao);	// get 1 vao id

	glGenVertexArrays(1, &vaoLines);	// get 1 vaoLines id


	glGenBuffers(4, vbo);	// Generate 1 buffer		// removed & becuase its an array now & 2 buffer now
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	//float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };

	int j = 0;
	for (int i = 0; i < GRAPHPOINTS * 3 * 2 * 2; i += 12)		//generating graphpoints
	{
		float randX1 = (rand() % 100 - 50) / 100.0;
		float randY1 = (rand() % 100 - 50) / 100.0;
		vec3 rand1 = calcVec3withW(randX1, randY1);		// a random point on which a generated point is being mirrored


		float randX2 = (rand() % 100 - 50) / 100.0;
		float randY2 = (rand() % 100 - 50) / 100.0;
		vec3 rand2 = calcVec3withW(randX2, randY2);		// a random point on which a generated point is being mirrored

		vec3 kozep = calcVec3withW(0, 0);				// new point in the middle 

		kozep = getMirrorOnPoint(kozep, rand1);			// new point being translated elsewhere (on the hypebolic plane) randomly
		kozep = getMirrorOnPoint(kozep, rand2);
		verticesCoordinates[j] = kozep.x;
		verticesCoordinates[j + 1] = kozep.y;
		j += 2;

		// the four corners of the square (six points in total becasuse of the 2 triangles) being translated
		vec3 egy = calcVec3withW(-SIDELENGTH / 2, -SIDELENGTH / 2);
		vec3 ketto = calcVec3withW(-SIDELENGTH / 2, +SIDELENGTH / 2);
		vec3 harom = calcVec3withW(+SIDELENGTH / 2, -SIDELENGTH / 2);
		vec3 negy = calcVec3withW(+SIDELENGTH / 2, +SIDELENGTH / 2);
		egy = getMirrorOnPoint(egy, rand1);
		egy = getMirrorOnPoint(egy, rand2);
		ketto = getMirrorOnPoint(ketto, rand1);
		ketto = getMirrorOnPoint(ketto, rand2);
		harom = getMirrorOnPoint(harom, rand1);
		harom = getMirrorOnPoint(harom, rand2);
		negy = getMirrorOnPoint(negy, rand1);
		negy = getMirrorOnPoint(negy, rand2);


		vertices[i + 0] = egy.x;
		vertices[i + 1] = egy.y;

		vertices[i + 2] = ketto.x;
		vertices[i + 3] = ketto.y;

		vertices[i + 4] = harom.x;
		vertices[i + 5] = harom.y;

		vertices[i + 6] = ketto.x;
		vertices[i + 7] = ketto.y;
		vertices[i + 8] = harom.x;
		vertices[i + 9] = harom.y;

		vertices[i + 10] = negy.x;
		vertices[i + 11] = negy.y;
	}

	for (int i = 0; i < GRAPHPOINTS * 2; i += 2)
	{
		printf("X:%3.2f Y:%3.2f\n", verticesCoordinates[i], verticesCoordinates[i + 1]);
	}

	j = 0;
	for (int i = 0; i < NUMOFEDGES * 2; i += 4) {
		int randA = rand() % GRAPHPOINTS;
		int randB;
		do {
			randB = rand() % GRAPHPOINTS;
		} while (randA == randB);
		/*verticesLines[i + 0] = verticesCoordinates[randA * 2 + 0];
		verticesLines[i + 1] = verticesCoordinates[randA * 2 + 1];*/
		//graphEdges[j + 0] = vec2(verticesLines[i + 0], verticesLines[i + 1]);
		adjacencyMtx[randA][randB] = true;

		/*verticesLines[i + 2] = verticesCoordinates[randB * 2 + 0];
		verticesLines[i + 3] = verticesCoordinates[randB * 2 + 1];*/
		//graphEdges[j + 1] = vec2(verticesLines[i + 2], verticesLines[i + 3]);
		adjacencyMtx[randB][randA] = true;
		//j += 2;
	}
	// TODO késõbb átrakni másik fv-be ha van idõ
	int n = 0;
	for (int i = 0; i < GRAPHPOINTS; i++) {
		for (int j = i + 1; j < GRAPHPOINTS; j++) {
			if (adjacencyMtx[i][j]) {
				verticesLines[n + 0] = verticesCoordinates[i * 2];
				verticesLines[n + 1] = verticesCoordinates[i * 2 + 1];

				verticesLines[n + 2] = verticesCoordinates[j * 2];
				verticesLines[n + 3] = verticesCoordinates[j * 2 + 1];
				n += 4;
			}
		}
	}

	for (int i = 0; i < GRAPHPOINTS * 2 * 2 * 3; i += 12) {
		textureCoordinates[i + 0] = { -1.0f };
		textureCoordinates[i + 1] = { -1.0f };

		textureCoordinates[i + 2] = { -1.0f };
		textureCoordinates[i + 3] = { +1.0f };

		textureCoordinates[i + 4] = { +1.0f };
		textureCoordinates[i + 5] = { -1.0f };

		textureCoordinates[i + 6] = { -1.0f };
		textureCoordinates[i + 7] = { +1.0f };

		textureCoordinates[i + 8] = { +1.0f };
		textureCoordinates[i + 9] = { -1.0f };

		textureCoordinates[i +10] = { +1.0f };
		textureCoordinates[i +11] = { +1.0f };
	}

	for (int i = 0; i < GRAPHPOINTS * 3 * 2; i+=6){
		colorSeeds[i + 0] = rand() % 200;
		colorSeeds[i + 1] = colorSeeds[i + 0];
		colorSeeds[i + 2] = colorSeeds[i + 0];
		colorSeeds[i + 3] = colorSeeds[i + 0];
		colorSeeds[i + 4] = colorSeeds[i + 0];
		colorSeeds[i + 5] = colorSeeds[i + 0];
	}


	glBindVertexArray(vao);		// make it active
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later
	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed


	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);	// binding texture vbo
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(textureCoordinates),  // # bytes
		textureCoordinates,	      	// address
		GL_STATIC_DRAW);	// we do not change later
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1,       // vbo -> AttribArray 1
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed


	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(colorSeeds),  // # bytes
		colorSeeds,	      	// address
		GL_STATIC_DRAW);	// we do not change later
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2,       // vbo -> AttribArray 2
		1, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	glBindVertexArray(vaoLines);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);	// binding lines vbo
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(verticesLines),  // # bytes
		verticesLines,	      	// address
		GL_STATIC_DRAW);	// we do not change later
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,
		2, GL_FLOAT, GL_FALSE,
		0, NULL);
	

	//glVertexAttribPointer(0,       // vbo -> AttribArray 0
	//	2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
	//	0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	//	in case of motion
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	graphMoves();		// kulon fv-be vittem a graf mozgatast

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later			//DYNAMIC DRAWRA (fent is a textúrán kívül a többit)

	//TODO valami

	glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(verticesLines),  // # bytes
		verticesLines,	      	// address
		GL_STATIC_DRAW);	// we do not change later


	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 1.0f, 0.8f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location


	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 * GRAPHPOINTS * 2 /*# Elements*/);		// 3 for 3 points * graph points * 2 triangles each


	glBindVertexArray(vaoLines);
	glDrawArrays(GL_LINES, 0 /*startIdx*/, NUMOFEDGES * 2 /*# Elements*/);

	glutSwapBuffers(); // exchange buffers for double buffering
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == 32) { isSpace = true; }
	else isSpace = false;
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
	onDisplay();
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char *buttonStat;
	switch (state) {
		case GLUT_DOWN: buttonStat = "pressed"; mouseXPrev = cX; mouseYPrev = cY; mouseXNext = cX; mouseYNext = cY; break;				// Mouse state coordinates getting a default value, as to not be "choppy"
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
	if (isSpace) {
		centreOfMass = calcCentreOfMass(centreOfMass);		//kiszamolom a tomegkozeppontot
		//TODO:	kiszámolni a tömegközéppontot, annak az origoba mutató vektorát és minden csúcsot ennyivel eltolni minden iterációban
		vec3 v_centreMass = calcVec3withW(centreOfMass.x, centreOfMass.y);

		int j = 0;
		// gráf középpontjainak mozgatása
		for (int i = 0; i < GRAPHPOINTS; i += 2) {		//TODO: mozgatni majd a négyzet csúcsait is
			vec2 Fi = sumForce(i) - velocity[i] * FRICTION;
			velocity[i] = velocity[i] + Fi * 0.005f;		// last float is time since last drawing

			// Fi vektorral odébbtolni az adott koordinátát
			float x = verticesCoordinates[j];
			float y = verticesCoordinates[j + 1];
			vec3 temp = pointFromVDir(calcVec3withW(x, y), velocity[i], 0.0005f);		// más távolság

			verticesCoordinates[j] = temp.x;
			verticesCoordinates[j + 1] = temp.y;
			j += 2;
		}
		//onDisplay();
		//// gráf négyzet csúcsok mozgatása
		//j = 0;
		//int k = 0;
		//for (int i = 0; i < GRAPHPOINTS * 3 * 2 * 2; i += 2) {
		//	vec2 Fi = sumForce(i) - velocity[k] * FRICTION;
		//	//velocity[i] = velocity[i] + Fi * 0.05f;		// last float is time since last drawing

		//	// Fi vektorral odébbtolni az adott koordinátát
		//	float x = vertices[j];
		//	float y = vertices[j + 1];
		//	vec3 temp = pointFromVDir(vec3(x, y, calcW(x, y)), velocity[k], vec3Distance(vec3(x, y, calcW(x, y)), velocity[k]));
		//	vertices[j] = temp.x;
		//	vertices[j + 1] = temp.y;
		//	j += 2;
		//	k++;
		//}

		//// élek mozgatása
		//j = 0;
		//k = 0;
		//for (int i = 0; i < NUMOFEDGES * 4; i += 2) {
		//	vec2 Fi = sumForce(i) - velocity[k] * FRICTION;
		//	//velocity[i] = velocity[i] + Fi * 0.05f;		// last float is time since last drawing

		//	// Fi vektorral odébbtolni az adott koordinátát
		//	float x = verticesLines[j];
		//	float y = verticesLines[j + 1];
		//	vec3 temp = pointFromVDir(vec3(x, y, calcW(x, y)), velocity[k], vec3Distance(vec3(x, y, calcW(x, y)), velocity[k]));
		//	verticesLines[j] = temp.x;
		//	verticesLines[j + 1] = temp.y;
		//	j += 2;
		//	k++;
		//}

	}
}


// Gets distance between 2 vec3 vectors
float vec3Distance(vec3 p, vec3 q) {
	//printf("TAVOLSAG\n");
	//printVec3(q);
	//printVec3(p);
	//printf("TAVOLSAGvege\n");
	//printf("%3.5f", -(p.x * q.x + p.y * q.y - p.z * q.z));
	return acosh(-(q.x * p.x + q.y * p.y - q.z * p.z));
}

// Gets a point from a vec3, a direction vetor and a distance
vec3 pointFromVDir(vec3 p, vec3 v, float t) {
	return p * cosh(t) + v * sinh(t);
}

// Gets a direction vector from 2 points and a distance
vec3 dirVecFrom2Points(vec3 p, vec3 q, float d_pq) {
	return (q - p * cosh(d_pq)) / (sinh(d_pq));
}

// Gets Mn from a->b line equation
vec3 getDivider(vec3 a, vec3 b, float ratio) {
	float d_ab = vec3Distance(a, b);
	//printf("d_ab = %3.5f\n", d_ab);
	vec3 v = dirVecFrom2Points(a, b, d_ab);
	return (pointFromVDir(a, v, ratio * d_ab));
}

// Mirrors one time
vec3 getMirrorOnPoint(vec3 p, vec3 on) {
	return getDivider(p, on, 2);
}

// prints a vec3 to console
void printVec3(vec3 in) {
	printf("X: %3.2f Y:%3.2f W:%3.2f\n", in.x, in.y, in.z);
}

// calculates w from given X and Y coordinates
float calcW(float x, float y) {
	return (sqrtf(x * x + y * y + 1));
}

vec3 calcHyperbolicCoord(float x, float y) {
	return  vec3(x, y, 1) / sqrtf(1 - x * x - y * y);
}

vec3 calcVec3withW(float x, float y) {
	return vec3(x, y, calcW(x, y));
}


vec2 sumForce(int idxX) {	//a pont idx-e (x koordinata)
	vec2 summaF = { 0.0f, 0.0f };
	vec2 currPoint = { verticesCoordinates[idxX], verticesCoordinates[idxX + 1] };

	int idx = idxX / 2;
	// loop hogy melyikkel van párban
	for (int i = 0; i < GRAPHPOINTS; i++) {
		// ha párban van
		if (adjacencyMtx[idx][i]) {
			// milyen messze --> annak fv-ben hogy közelebb távolabb változik
			vec3 current = calcVec3withW(verticesCoordinates[idxX], verticesCoordinates[idxX + 1]);
			vec3 other = calcVec3withW(verticesCoordinates[i * 2], verticesCoordinates[i * 2 + 1]);
			float distance = vec3Distance(current, other);
			// summaF += pozitiv/negativ ero ;
			if (distance < optimalDistance) summaF = summaF + 0.5f;
			else summaF = summaF - 0.5f;
		}
		// ha nincs párban
		else {
			// milyen messze --> annak fv-ben hogy közelebb távolabb változik
			vec3 current = calcVec3withW(verticesCoordinates[idxX], verticesCoordinates[idxX + 1]);
			vec3 other = calcVec3withW(verticesCoordinates[i * 2], verticesCoordinates[i * 2 + 1]);
			float distance = vec3Distance(current, other);
			// summaF -= pozitiv/negativ ero ;
			if (distance < optimalDistance) summaF = summaF - 0.5f;
			else summaF = summaF + 0.5f;
		}
	}
	return summaF;
}

vec2 calcCentreOfMass(vec2 in) {
	for (int i = 0; i < GRAPHPOINTS * 2; i += 2) {
		in = in + vec2(verticesCoordinates[i + 0], verticesCoordinates[i + 1]);
	}
	in = in / GRAPHPOINTS;
	return in;
}

void graphMoves() {
	if (mouseXNext != mouseXPrev || mouseYNext != mouseYPrev) {		// as to not get zero division		todo: check if on hyperbolic plane
		if (mouseXNext * mouseXNext + mouseYNext * mouseYNext < 1) {
			vec3 a = calcHyperbolicCoord(mouseXPrev, mouseYPrev);
			printf("mouseXPrev: %3.5f, mouseYPrev: %3.5f\n", mouseXPrev, mouseYPrev);
			printVec3(a);
			vec3 b = calcHyperbolicCoord(mouseXNext, mouseYNext);
			printVec3(b);
			vec3 m1 = getDivider(a, b, 0.25f);
			vec3 m2 = getDivider(a, b, 0.75f);

			// a graf kozeppontjainak eltolasa
			int l = 0;
			for (int i = 0; i < GRAPHPOINTS * 2; i += 2) {
				vec3 temp = { verticesCoordinates[i + 0], verticesCoordinates[i + 1], calcW(verticesCoordinates[i + 0], verticesCoordinates[i + 1]) };
				//printVec3(temp);
				temp = getMirrorOnPoint(temp, m1);
				temp = getMirrorOnPoint(temp, m2);
				//printVec3(temp);
				verticesCoordinates[i + 0] = temp.x;			// X coordinate of said point
				verticesCoordinates[i + 1] = temp.y;			// Y coordinate of said point

				// the four corners of the square (six points in total becasuse of the 2 triangles) being translated
				vec3 origo = { 0.0f, 0.0f, 1.0f };
				vec3 pont = calcVec3withW(verticesCoordinates[i + 0], verticesCoordinates[i + 1]);
				vec3 m1_2 = getDivider(origo, pont, 0.25f);
				vec3 m2_2 = getDivider(origo, pont, 0.75f);
				vec3 egy = calcVec3withW(-SIDELENGTH / 2, -SIDELENGTH / 2);
				vec3 ketto = calcVec3withW(-SIDELENGTH / 2, +SIDELENGTH / 2);
				vec3 harom = calcVec3withW(+SIDELENGTH / 2, -SIDELENGTH / 2);
				vec3 negy = calcVec3withW(+SIDELENGTH / 2, +SIDELENGTH / 2);
				egy = getMirrorOnPoint(egy, m1_2);
				egy = getMirrorOnPoint(egy, m2_2);
				ketto = getMirrorOnPoint(ketto, m1_2);
				ketto = getMirrorOnPoint(ketto, m2_2);
				harom = getMirrorOnPoint(harom, m1_2);
				harom = getMirrorOnPoint(harom, m2_2);
				negy = getMirrorOnPoint(negy, m1_2);
				negy = getMirrorOnPoint(negy, m2_2);

				vertices[l + 0] = egy.x;
				vertices[l + 1] = egy.y;

				vertices[l + 2] = ketto.x;
				vertices[l + 3] = ketto.y;

				vertices[l + 4] = harom.x;
				vertices[l + 5] = harom.y;

				vertices[l + 6] = ketto.x;
				vertices[l + 7] = ketto.y;
				vertices[l + 8] = harom.x;
				vertices[l + 9] = harom.y;

				vertices[l + 10] = negy.x;
				vertices[l + 11] = negy.y;
				l += 12;
			}

			int n = 0;
			for (int i = 0; i < GRAPHPOINTS; i++) {
				for (int j = i + 1; j < GRAPHPOINTS; j++) {
					if (adjacencyMtx[i][j]) {
						verticesLines[n + 0] = verticesCoordinates[i * 2];
						verticesLines[n + 1] = verticesCoordinates[i * 2 + 1];

						verticesLines[n + 2] = verticesCoordinates[j * 2];
						verticesLines[n + 3] = verticesCoordinates[j * 2 + 1];
						n += 4;
					}
				}
			}
		}
	}
}

vec2 calcMovingVector(float x, float y, vec3 m1, vec3 m2) {
	vec3 temp = { x, y, calcW(x, y) };
	//printVec3(temp);
	temp = getMirrorOnPoint(temp, m1);
	temp = getMirrorOnPoint(temp, m2);
	//printVec3(temp);
	return vec2(temp.x, temp.y);
}

bool isVec2Equal(vec2 a, vec2 b) {
	return (a.x == b.x && a.y == b.y);
}

