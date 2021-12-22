#include "Simulation.h"


__host__ __device__ void calculateParticleCenter(Particle*		particles,
												 ParticleNode*	nodes,
												 unsigned int	particle_index)
{
	// https://math.stackexchange.com/questions/1592128/finding-center-of-mass-for-tetrahedron

	// Zero the current barycenter
	particles[particle_index].barycenter = make_double3(0.0, 0.0, 0.0);

	// Go through the particle nodes
	for (int j = 0; j < 4; j++)
	{
		// Add the node positions to the barycenter
		particles[particle_index].barycenter += nodes[particles[particle_index].nodes[j]].position;
	}

	// Divide the coordinates of the barycenter by 4
	particles[particle_index].barycenter *= 0.25;
}



__host__ void assignParticleMaterial(Particle* particles,
									 unsigned int	particle_index,
									 MaterialProperty* materials,
									 unsigned int size_of_materials)
{
	// Create a variable with infinite distance
	double min_distance = INF;

	// Create a variable for the material index
	int material_index = -1;

	// Go through the materials and find the closest one to the barycenter of the particle
	for (int j = 0; j < size_of_materials; j++)
	{
		// Calculate the distance between the barycenter of the particle and location of the material property
		double distance = length(materials[j].location - particles[particle_index].barycenter);

		// Check if the current material is closer to the particle the current minimum
		if (distance < min_distance)
		{
			min_distance = distance;
			material_index = j;
		}
	}

	// Assign the closest material to the particle
	particles[particle_index].material_type = materials[material_index].type;
	particles[particle_index].material_property = material_index;
	
}



__host__ __device__ void calculateParticleVolume(Particle*		particles,
												 ParticleNode*	nodes,
												 unsigned int	particle_index)
{
	// https://www.vedantu.com/question-answer/find-the-volume-of-tetrahedron-whose-vertices-class-10-maths-cbse-5eedf688ccf1522a9847565b

	// Get the node positions
	// Node A -> 0; Node B -> 1; Node C -> 2; Node D -> 3
	double3 node_a_position = nodes[particles[particle_index].nodes[0]].position;
	double3 node_b_position = nodes[particles[particle_index].nodes[1]].position;
	double3 node_c_position = nodes[particles[particle_index].nodes[2]].position;
	double3 node_d_position = nodes[particles[particle_index].nodes[3]].position;

	// Calculate the edge vectors
	double3 vector_ab = node_b_position - node_a_position;
	double3 vector_ac = node_c_position - node_a_position;
	double3 vector_ad = node_d_position - node_a_position;

	// Calculate the cross product of AB & AC
	double3 cross_product_ab_ac = cross(vector_ab, vector_ac);

	// Calculate the dot product of AD and the cross product of AB & AC
	double volume_parallelepipedon = dot(vector_ad, cross_product_ab_ac);

	// Calculate the volume of the tetrahedron
	particles[particle_index].volume = fabs(volume_parallelepipedon / 6.0);
}


__host__ void calculateParticleMass(Particle*			particles,
									MaterialProperty*	materials,
									unsigned int		particle_index)
{
	// Get the density
	double density = materials[particles[particle_index].material_property].density;

	// Calculate the mass
	particles[particle_index].mass = particles[particle_index].volume * density;
}




__host__ __device__ void calculateParticleCircumsphere(Particle			*particles, 
													   ParticleNode		*nodes, 
													   unsigned int		particle_index)
{
	// Reset the circumsphere radius
	particles[particle_index].circumsphere_radius = 0.0;

	// Go through the particle nodes
	for (int j = 0; j < 4; j++)
	{
		// Position of the current node
		double3 node_position = nodes[particles[particle_index].nodes[j]].position;
			
		// Distance from the barycenter to the current node
		double distance = length(node_position - particles[particle_index].barycenter);

		// Check if the distance is larger the current circumsphere radius
		if (distance > particles[particle_index].circumsphere_radius)
		{
			// Assign it to the particle
			particles[particle_index].circumsphere_radius = distance;
		}
	}
}



__host__ __device__ void checkParticleDamage(Particle*			particles,
											 unsigned int		number_of_particles,
											 ParticleFace*		faces,
											 IntersectionNode*	intersection_nodes,
											 AxialSpring*		axial_springs,
											 RotationalSpring*	rotational_springs)
{
	// Go through the particles
	for (int i = 0; i < number_of_particles; i++)
	{
		// Go through the axial springs
		for (int j = 0; j < 3; j++)
		{
			// Get the index of the spring
			int spring_index = particles[i].axial_springs[j];

			// Get the total force
			double3 total_force = axial_springs[spring_index].total_force_node_a;

			// Calculate the magnitude of the total force
			double total_force_magnitude = length(total_force);

			// Get the strength of the spring
			double strength = axial_springs[spring_index].strength;

			// Check if the magnitude of the total force is larger than the strength of the spring
			if (total_force_magnitude > strength)
			{
				// The particle is damaged

				// Set the particle status to be damaged
				particles[i].status = 2;

				// Set the faces to be damaged
				for (int k = 0; k < 4; k++)
				{
					faces[particles[i].faces[k]].status = 2;
				}

				// Set the intersection nodes to be damaged
				for (int k = 0; k < 6; k++)
				{
					intersection_nodes[particles[i].intersections[k]].status = 2;
				}

				// Set the axial springs to be damaged
				for (int k = 0; k < 3; k++)
				{
					axial_springs[particles[i].axial_springs[k]].status = 2;
				}

				// Set the rotational springs to be damaged
				for (int k = 0; k < 3; k++)
				{
					rotational_springs[particles[i].rotational_springs[k]].status = 2;
				}

				// Break the loop
				break;
			}
		}
	}
}


__host__ void initializeParticles(Particle*			particles,
								  unsigned int		number_of_particles,
								  ParticleNode*		nodes,
								  MaterialProperty* materials,
								  unsigned int		size_of_materials,
								  Settings&			settings)
{

	std::vector<std::thread> threads;

	auto initialize = [](Particle* particles,
						 const int			number_of_particles,
						 ParticleNode*		nodes,
						 MaterialProperty*	materials,
						 unsigned int		size_of_materials,
						 const int			number_of_threads,
						 const int			thread_id)
	{
		int particle_index = thread_id;
		while (particle_index < number_of_particles)
		{
			calculateParticleCenter(particles, nodes, particle_index);
			assignParticleMaterial(particles, particle_index, materials, size_of_materials);
			calculateParticleVolume(particles, nodes, particle_index);
			calculateParticleMass(particles, materials, particle_index);
			calculateParticleCircumsphere(particles, nodes, particle_index);
			calculateParticleNodeMass(particles, nodes, particle_index);
			particle_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < settings.number_of_CPU_threads; thread_id++)
	{
		threads.push_back(std::thread(initialize,
									  particles,
									  number_of_particles,
									  nodes,
									  materials,
									  size_of_materials,
									  settings.number_of_CPU_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}


__host__ void updateParticlesCPU(Particle*		particles,
								 unsigned int	number_of_particles,
								 ParticleNode*	nodes,
								 unsigned int	number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](Particle*		particles,
					 const int		number_of_particles,
					 ParticleNode*	nodes,
					 const int		number_of_threads,
					 const int		thread_id)
	{
		int particle_index = thread_id;
		while (particle_index < number_of_particles)
		{
			if (particles[particle_index].status == 1)
			{
				calculateParticleCenter(particles, nodes, particle_index);
				calculateParticleVolume(particles, nodes, particle_index);
				calculateParticleCircumsphere(particles, nodes, particle_index);
			}

			particle_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  particles,
									  number_of_particles,
									  nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}