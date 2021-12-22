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
		// Assign the mass to the node
		nodes[particles[particle_index].nodes[j]].mass = nodal_mass;
	}
}




__host__ __device__ void integrateParticleNode(ParticleNode*	nodes,
											   unsigned int		node_index,
											   const double		timestep)
{
	nodes[node_index].acceleration = nodes[node_index].force / nodes[node_index].mass;
	nodes[node_index].velocity += nodes[node_index].acceleration * timestep;
	nodes[node_index].position += nodes[node_index].velocity * timestep;
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
