#include "ParticleSystem.h"


ParticleSystem::ParticleSystem(const unsigned int const _MAX_PARTICLES) {
	MAX_PARTICLES = _MAX_PARTICLES;
	// Quad vertices
	GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, 0.5f, 0.0f,
		0.5f, 0.5f, 0.0f,
	};
	g_particule_position_size_data = new GLfloat[MAX_PARTICLES * 4];

	nodeContainer = new OctreeNode[4 * MAX_PARTICLES];


	// Create share
	ParticlesContainer = new Particle[_MAX_PARTICLES];
	initParticleSystem();
	
	
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// The VBO containing the quad/cube
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// The VBO containing the positions and sizes of the particles
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 4 * sizeof(GLfloat), g_particule_position_size_data, GL_STREAM_DRAW);


	// Generate buffers for the box
	glGenVertexArrays(1, &BoxVAO);
	glGenBuffers(1, &BoxVBO);
	glGenBuffers(1, &BoxEBO);

	// Generate buffers for com triangle
	glGenVertexArrays(1, &comVAO);
	glGenBuffers(1, &comVBO);


}

void ParticleSystem::renderCOM(OctreeNode *node, Shader comShader){

	GLfloat triangle_vertices[] = {
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		0.0f, 0.5f, 0.0f
	};



	glm::mat4 model(1.0f);
	model = glm::translate(model, glm::vec3(node->com_x, node->com_y, node->com_z));
	model = glm::scale(model, glm::vec3(4.0f, 4.0f, 4.0f));
	comShader.setMat4("model", model);


	glBindVertexArray(comVAO);

	glBindBuffer(GL_ARRAY_BUFFER, comVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_vertices), triangle_vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glBindVertexArray(comVAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void ParticleSystem::initParticleSystem(){

	float phi, r, x, y, z;

	// Populate initial positions
	for (int i = 0; i < MAX_PARTICLES; i++) {

		Particle p = ParticlesContainer[i];
		

		

		phi = (float) rand() / RAND_MAX * 2.0f * M_PI;
		r = (float) rand() / RAND_MAX * MAX_DISTANCE;

		

		float x = r * cos(phi);
		float y = 10.0f + r * sin(phi);
		float z = (rand() % (2 * 40) - (float)40);


		// Setup particle
		p.weight = 500000.0f;
		p.px = x;
		p.py = y;
		p.pz = z;
		
		
		if (i % 2 == 0){
			p.px += 200;
		}
		else {
			p.px -= 200;
		}


		glm::vec3 speed = 40.0f * glm::cross(-glm::vec3(p.px, p.py, p.pz), glm::vec3(0.0, 0.0, 1.0));
		
		p.vx = speed.x;
		p.vy = speed.y;
		p.vz = speed.z;

		
		/* DEBUG PLACEMENT */
		/*
		// Place particles in a single ring, easier to see if force calculation looks correct
		r = 100.0f + (float) rand() / RAND_MAX * 0.01 * MAX_DISTANCE;

		x = r * cos(phi);
		y = 10.0f + r * sin(phi);
		z = 0;

		p.px = x;
		p.py = y;
		p.pz = z;

		p.vx = 0.0f;
		p.vy = 0.0f;
		p.vz = 0.0f;
		*/


		ParticlesContainer[i] = p;

		// Setup position in buffer
		g_particule_position_size_data[i * 3 + 0] = p.px;
		g_particule_position_size_data[i * 3 + 1] = p.py;
		g_particule_position_size_data[i * 3 + 2] = p.pz;
	}
}

ParticleSystem::~ParticleSystem(){
	delete[] g_particule_position_size_data;
	delete[] ParticlesContainer;
}

void ParticleSystem::getBounds(float &_minx, float &_maxx, float &_miny, float &_maxy, float &_minz, float &_maxz){


	float minx = ParticlesContainer[0].px;
	float maxx = ParticlesContainer[0].px;
	
	float miny = ParticlesContainer[0].py;
	float maxy = ParticlesContainer[0].py;

	float minz = ParticlesContainer[0].pz;
	float maxz = ParticlesContainer[0].pz;

	for (int i = 0; i < MAX_PARTICLES; i++){
		glm::vec3 pos = glm::vec3(ParticlesContainer[i].px, ParticlesContainer[i].py, ParticlesContainer[i].pz);
		
		minx = std::min(pos.x, minx);
		maxx = std::max(pos.x, maxx);

		miny = std::min(pos.y, miny);
		maxy = std::max(pos.y, maxy);

		minz = std::min(pos.z, minz);
		maxz = std::max(pos.z, maxz);
	}


	_minx = minx;
	_maxx = maxx;

	_miny = miny;
	_maxy = maxy;

	_minz = minz;
	_maxz = maxz;
	
}

void ParticleSystem::renderBounds(Shader boxShader){

	float vertices[] = {
		1.0f, 1.0f, 1.0f,  // front top right		0
		1.0f, -1.0f, 1.0f,  // front bottom right	1
		-1.0f, -1.0f, 1.0f,  // front bottom left	2
		-1.0f, 1.0f, 1.0f,   // front top left		3

		1.0f, 1.0f, -1.0f,  // back top right		4
		1.0f, -1.0f, -1.0f,  // back bottom right	5
		-1.0f, -1.0f, -1.0f,  // back bottom left	6
		-1.0f, 1.0f, -1.0f   // back top left		7
	};
	unsigned int indices[] = {  // note that we start from 0!
		2, 1, 0,
		3, 2, 6,
		5, 1, 0,
		4, 5, 6,
		7, 3, 7,
		4

	};

	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(BoxVAO);
	glBindBuffer(GL_ARRAY_BUFFER, BoxVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, BoxEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glm::mat4 model(1.0f);
	for (int i = 0; i < 10; i++){
		root->setModelAndRender(boxShader);
	}
	
	glDisableVertexAttribArray(0);


}

void ParticleSystem::calcTreeCOM(OctreeNode *node){
	if (!node) return;

	// If node is a leaf the mass and COM are the mass and COM of the data
	if (node->getNoElements() == 1) return;

	// If node is an empty cell
	else {

		
		node->m = 0.0f; 
		node->com_x = 0.0f; 
		node->com_y = 0.0f; 
		node->com_z = 0.0f;

		// Step through the childrens to add their mass to this node
		for (int i = 0; i < 8; i++){
			if (node->getChild(i) != nullptr){
				calcTreeCOM(node->getChild(i));

				OctreeNode *child = node->getChild(i);

				//float child_mass = child_pt->mass;
				node->m += child->m;
				node->com_x += child->m*(child->com_x);
				node->com_y += child->m*(child->com_y);
				node->com_z += child->m*(child->com_z);
			}
		}
		node->com_x /= node->m;
		node->com_y /= node->m;
		node->com_z /= node->m;
	}
}

void ParticleSystem::buildTree(){

	

	float minx, miny, minz, maxx, maxy, maxz;
	getBounds(minx, maxx, miny, maxy, minz, maxz);

	root = new OctreeNode(minx, miny, minz, maxx, maxy, maxz);


	for (int i = 0; i < MAX_PARTICLES; i++){
		Particle p = ParticlesContainer[i];

		
		root->insert(p.px, p.py, p.pz, p.weight, p.px, p.py, p.pz);
	}
}

void ParticleSystem::flattenTree(OctreeNode *node, int &count){
	
	count = 0;
	std::queue<OctreeNode*> q;

	if (node) {
		q.push(node);
	}


	while (!q.empty()) {
		OctreeNode * temp_node = q.front();
		
		temp_node->index = count;
		nodeContainer[count] = *temp_node;
		count++;
		q.pop();
		
		for (int i = 0; i < 8; i++){
			if (temp_node->getChild(i)) {
				q.push(temp_node->getChild(i));
			}
		}
	}


	for (int j = 0; j < count; j++){
		for (int i = 0; i < 8; i++){

			if (nodeContainer[j].children[i]){
				nodeContainer[j].childIndices[i] = nodeContainer[j].children[i]->index;
			}
			else {
				nodeContainer[j].childIndices[i] = NULL;
			}
		}
	}
}

void ParticleSystem::render(float dt){
	
	std::chrono::high_resolution_clock::time_point start, tBuildTree, tCalcCOM, tFlatten, tCalcForces, tUpdPos;

	// Start timer
	start = MyTimer::getTime();

	buildTree();
	tBuildTree = MyTimer::getTime(); // get time

	calcTreeCOM(root);
	tCalcCOM = MyTimer::getTime(); // get time

	int count = 0;
	
	flattenTree(root, count);
	tFlatten = MyTimer::getTime(); // get time

	CUDACalcForces(ParticlesContainer, nodeContainer, count, MAX_PARTICLES, dt);
	tCalcForces = MyTimer::getTime();

	CUDAUpdatePositions(ParticlesContainer, g_particule_position_size_data, MAX_PARTICLES, dt);
	tUpdPos = MyTimer::getTime();

	std::cout << "Build tree: \t" << MyTimer::getDeltaTimeMS(start, tBuildTree) << std::endl;
	std::cout << "Compu. COM: \t" << MyTimer::getDeltaTimeMS(tBuildTree, tCalcCOM) << std::endl;
	std::cout << "Flat. tree: \t" << MyTimer::getDeltaTimeMS(tCalcCOM, tFlatten) << std::endl;
	std::cout << "Cal. force: \t" << MyTimer::getDeltaTimeMS(tFlatten, tCalcForces) << std::endl;
	std::cout << "Upd. posit: \t" << MyTimer::getDeltaTimeMS(tCalcForces, tUpdPos) << std::endl << std::endl;
	



	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, MAX_PARTICLES * sizeof(GLfloat) * 4, g_particule_position_size_data);

	
	// Setup attributes for the particle shader
	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	

	// 2nd attribute buffer : positions of particles' centers
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
	glVertexAttribDivisor(1, 1); // positions : one per quad (its center)                 -> 1

	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, MAX_PARTICLES);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

void ParticleSystem::calcParticleForce(Particle &p, OctreeNode *node, float dt){

	
	if (!node) return ;

	
	float dx = node->com_x - p.px;
	float dy = node->com_y - p.py;
	float dz = node->com_z - p.pz;

	float dist = sqrt(dx*dx + dy*dy + dz*dz);

	if (dist == 0) return;

	float width = ((node->max_x - node->min_x) +
		(node->max_y - node->min_y) +
		(node->max_z - node->min_z)) / 3;

	// The node is far away enough to be evaluated as a single node
	if (width / dist < 0.5){

		float F = (G * p.weight * node->m) / (dist + 1.0f + SOFTENING * SOFTENING);

		p.vx += F * dx / dist;
		p.vy += F * dy / dist;
		p.vz += F * dz / dist;
	} 
	
	// The node is to close to be treated as a single particle and must be further traversed
	else {
		for (int i = 0; i < 8; i++){
			calcParticleForce(p, node->getChild(i), dt);
		}
	}	
	
}

void ParticleSystem::BarnesHutUpdateForces(float dt){
	
	
	for (int i = 0; i < MAX_PARTICLES; i++){

		Particle p = ParticlesContainer[i];
		calcParticleForce(p, root, dt);
		ParticlesContainer[i] = p;
	}
	

	//updateForces(dt);
	
}

// These functions should launch the kernels for the respective framework
void ParticleSystem::updateForces(float dt){

	for (int i = 0; i < MAX_PARTICLES; i++) {

		Particle pi = ParticlesContainer[i];
		float Fx = 0; float Fy = 0; float Fz = 0;

		for (int j = 0; j < MAX_PARTICLES; j++){

			if (i != j){
				Particle pj = ParticlesContainer[j];

				float dx = pj.px - pi.px;
				float dy = pj.py - pi.py;
				float dz = pj.pz - pi.pz;

				float dist = sqrt(dx*dx + dy*dy + dz*dz);

				float F = (9.82 * pi.weight * pj.weight) / (dist + SOFTENING * SOFTENING);

				Fx += F * dx / dist;
				Fy += F * dy / dist;
				Fz += F * dz / dist;
			}
		}


		// Update speed
		pi.vx += Fx;
		pi.vy += Fy;
		pi.vz += Fz;

		ParticlesContainer[i] = pi;
	}

}

void ParticleSystem::updatePositions(float dt){

	float simspeed = 0.001f;


	for (int i = 0; i < MAX_PARTICLES; i++){
		Particle p = ParticlesContainer[i];

		p.px = p.px + p.vx * simspeed*dt;
		p.py = p.py + p.vy * simspeed*dt;
		p.pz = p.pz + p.vz * simspeed*dt;


		ParticlesContainer[i] = p;

		// Update position buffer
		g_particule_position_size_data[i * 3 + 0] = p.px;
		g_particule_position_size_data[i * 3 + 1] = p.py;
		g_particule_position_size_data[i * 3 + 2] = p.pz;
	}
}

