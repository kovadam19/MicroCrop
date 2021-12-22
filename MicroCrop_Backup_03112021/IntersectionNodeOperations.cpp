#include "Simulation.h"

__host__ void createParticleIntersectionNodes(IntersectionNodeContainer&	intersection_nodes,
											  ParticleNodeContainer&		nodes,
											  ParticleFaceContainer&		faces,
											  ParticleContainer&			particles,
											  MaterialContainer&			materials)
{
	// Go through all the particles
	for (int i = 0; i < particles.size(); i++)
	{
		// Get the particle material index
		int material_index = particles[i].material_property;

		// Number of intersection points per particle
		int nodes_per_particle = 0;

		// Go through the anisotropy axes
		for (int j = 0; j < 3; j++)
		{
			// Get the current axis
			double3 axis = materials[material_index].axes[j];

			// Number of intersection nodes
			int nodes_per_axis = 0;

			// Go through the faces
			for (int k = 0; k < 4; k++)
			{
				// Get the face index
				int face_index = particles[i].faces[k];

				// Get the position of one of the nodes on the face
				double3 node_a_position = nodes[faces[face_index].nodes[0]].position;

				// Calculate a vector from the particle barycenter to one of the nodes on the face
				double3 center_to_node = node_a_position - particles[i].barycenter;

				// Get the face normal
				double3 face_normal = faces[face_index].normal;

				// Calculate the normal component to the face normal
				double vector_normal_component = dot(center_to_node, face_normal);

				// Calculate the normal component of the axis to the face normal
				double axis_normal_component = dot(axis, face_normal);

				// Calculate the distance from the barycenter to the intersection point
				double distance = vector_normal_component / axis_normal_component;

				// Calculate the intersection point position
				double3 intersection_point = particles[i].barycenter + axis * distance;

				// Check if the point falls within the triangle
				double coefficients[3] = { 0.0, 0.0, 0.0 };
				if (pointOnParticleFace(&faces[0], &nodes[0], coefficients, particles[i].faces[k], intersection_point))
				{
					// Create a new intersection node
					IntersectionNode new_node;

					// Assign an ID to the intersection node
					new_node.id = _intersection_node_id++;

					// Assign the nodes to the intersection node
					new_node.nodes[0] = faces[face_index].nodes[0];
					new_node.nodes[1] = faces[face_index].nodes[1];
					new_node.nodes[2] = faces[face_index].nodes[2];

					// Assign the coefficients for the shape function
					new_node.coefficients[0] = coefficients[0];
					new_node.coefficients[1] = coefficients[1];
					new_node.coefficients[2] = coefficients[2];

					// Add the new node to the intersection node container
					intersection_nodes.push_back(new_node);

					// Add the intersection node to the particle
					particles[i].intersections[nodes_per_particle] = intersection_nodes.size() - 1;

					// Increase the number of intersection nodes
					nodes_per_axis++;
					nodes_per_particle++;

					// Check if we found all the nodes for the current axis
					if (nodes_per_axis == 2)
					{
						// Break the loop
						break;
					}
				}
			}
		}
	}
}


__host__ __device__ void calculateParticleIntersectionNodePosition(IntersectionNode*	intersection_nodes,
																   ParticleNode*		nodes,
																   unsigned int			inode_index)
{
	// Get the independent node positions
	double3 node_a_position = nodes[intersection_nodes[inode_index].nodes[0]].position;
	double3 node_b_position = nodes[intersection_nodes[inode_index].nodes[1]].position;
	double3 node_c_position = nodes[intersection_nodes[inode_index].nodes[2]].position;

	// Calculate the position of the the intersection node
	intersection_nodes[inode_index].position = intersection_nodes[inode_index].coefficients[0] * node_a_position +
											   intersection_nodes[inode_index].coefficients[1] * node_b_position +
											   intersection_nodes[inode_index].coefficients[2] * node_c_position;
}


__host__ __device__ void calculateParticleIntersectionNodeVelocity(IntersectionNode*	intersection_nodes,
																   ParticleNode*		nodes,
																   unsigned int			inode_index)
{
	// Get the independent node velocities
	double3 node_a_velocity = nodes[intersection_nodes[inode_index].nodes[0]].velocity;
	double3 node_b_velocity = nodes[intersection_nodes[inode_index].nodes[1]].velocity;
	double3 node_c_velocity = nodes[intersection_nodes[inode_index].nodes[2]].velocity;

	// Calculate the velocity of the the intersection node
	intersection_nodes[inode_index].velocity = intersection_nodes[inode_index].coefficients[0] * node_a_velocity +
											   intersection_nodes[inode_index].coefficients[1] * node_b_velocity +
											   intersection_nodes[inode_index].coefficients[2] * node_c_velocity;
}



__host__ void updateParticleIntersectionNodesCPU(IntersectionNode*	intersection_nodes,
												 unsigned int		number_of_inodes,
												 ParticleNode*		nodes,
												 unsigned int		number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](IntersectionNode*	intersection_nodes,
					 unsigned int		number_of_inodes,
					 ParticleNode*		nodes,
					 const int			number_of_threads,
					 const int			thread_id)
	{
		int inode_index = thread_id;
		while (inode_index < number_of_inodes)
		{
			calculateParticleIntersectionNodePosition(intersection_nodes, nodes, inode_index);
			calculateParticleIntersectionNodeVelocity(intersection_nodes, nodes, inode_index);
			inode_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  intersection_nodes,
									  number_of_inodes,
									  nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}