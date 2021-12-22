#include "Simulation.h"


__host__ void calculateParticleNodeMass(Particle*		particles,
										ParticleNode*	nodes,
										unsigned int	particle_index)
{
	// Calculate the nodal mass
	double nodal_mass = particles[particle_index].mass / 4.0;

	// Go through the particle nodes
	for (int j = 0; j < 4; j++)
	{
		// Add the mass to the node
		nodes[particles[particle_index].nodes[j]].mass += nodal_mass;
	}
}




__host__ __device__ int isParticleNodeOnPlane(ParticleNode* nodes,
											  unsigned int	node_index,
											  double3		location,
											  double3		normal)
{
	// Get the node position
	double3 node_position = nodes[node_index].position;

	// Calculate the distance vector
	double3 distance_vector = node_position - location;

	// Calculate the distance between the plane and the node
	double distance = dot(normal, distance_vector);

	// Check if the distance is zero
	if (isZero(distance))
	{
		return 1;
	}
	else 
	{
		return 0;
	}
}



__host__ __device__ void integrateParticleNode(ParticleNode*	nodes,
											   unsigned int		node_index,
											   const double		timestep)
{
	if (nodes[node_index].boundaries.x == 0)
	{
		nodes[node_index].acceleration.x = nodes[node_index].force.x / nodes[node_index].mass;
		nodes[node_index].velocity.x += nodes[node_index].acceleration.x * timestep;
		nodes[node_index].position.x += nodes[node_index].velocity.x * timestep;
	}
	else
	{
		nodes[node_index].acceleration.x = 0.0;
		nodes[node_index].velocity.x = 0.0;
	}

	if (nodes[node_index].boundaries.y == 0)
	{
		nodes[node_index].acceleration.y = nodes[node_index].force.y / nodes[node_index].mass;
		nodes[node_index].velocity.y += nodes[node_index].acceleration.y * timestep;
		nodes[node_index].position.y += nodes[node_index].velocity.y * timestep;
	}
	else
	{
		nodes[node_index].acceleration.y = 0.0;
		nodes[node_index].velocity.y = 0.0;
	}

	if (nodes[node_index].boundaries.z == 0)
	{
		nodes[node_index].acceleration.z = nodes[node_index].force.z / nodes[node_index].mass;
		nodes[node_index].velocity.z += nodes[node_index].acceleration.z * timestep;
		nodes[node_index].position.z += nodes[node_index].velocity.z * timestep;
	}
	else
	{
		nodes[node_index].acceleration.z = 0.0;
		nodes[node_index].velocity.z = 0.0;
	}
}


__host__ void resetParticleNodesCPU(ParticleNode*	nodes,
									unsigned int	number_of_nodes,
									unsigned int	number_of_threads)
{
	std::vector<std::thread> threads;

	auto reset = [](ParticleNode* nodes,
					unsigned int  number_of_nodes,
					const int	  number_of_threads,
					const int	  thread_id)
	{
		int node_index = thread_id;
		while (node_index < number_of_nodes)
		{
			nodes[node_index].force = make_double3(0.0, 0.0, 0.0);
			node_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(reset,
									  std::ref(nodes),
									  number_of_nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}



__host__ void updateParticleNodesCPU(ParticleNode*	nodes,
									 unsigned int	number_of_nodes,
									 unsigned int	number_of_threads,
									 double			timestep)
{
	std::vector<std::thread> threads;

	auto update = [](ParticleNode*	nodes,
					 unsigned int	number_of_nodes,
					 const double	timestep,
					 const int		number_of_threads,
					 const int		thread_id)
	{
		int node_index = thread_id;
		while (node_index < number_of_nodes)
		{
			integrateParticleNode(nodes, node_index, timestep);
			node_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  nodes,
									  number_of_nodes,
									  timestep,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}
