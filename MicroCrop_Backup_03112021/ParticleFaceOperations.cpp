#include "Simulation.h"


__host__ void createParticleFaces(ParticleFaceContainer& faces, ParticleContainer& particles, ParticleNodeContainer& nodes)
{
	// The first three points define the face (a-b-c) and the fourth one (d) helps to define the normal direction
	int faces_definition[4][4] = { {0, 1, 2, 3},
								   {0, 1, 3, 2},
								   {0, 2, 3, 1},
								   {1, 2, 3, 0} };

	// Loop through the particles
	for (int i = 0; i < particles.size(); i++)
	{
		// Loop through the faces
		for (int j = 0; j < 4; j++)
		{
			// Create a face
			ParticleFace new_face;

			// Assign an ID to the new face
			new_face.id = _face_id++;

			// Assign nodes to the face
			for (int k = 0; k < 4; k++)
			{
				new_face.nodes[k] = particles[i].nodes[faces_definition[j][k]];
			}

			// Add the new face to the container
			faces.push_back(new_face);

			// Assign the face to the particle
			particles[i].faces[j] = faces.size() - 1;
		}
	}
}




__host__ __device__ void calculateParticleFaceCenter(ParticleFace* faces,
													 ParticleNode* nodes,
													 unsigned int  face_index)
{
	// Get the node positions
	double3 node_a_position = nodes[faces[face_index].nodes[0]].position;
	double3 node_b_position = nodes[faces[face_index].nodes[1]].position;
	double3 node_c_position = nodes[faces[face_index].nodes[2]].position;

	// Calculate the barycenter
	faces[face_index].barycenter = (node_a_position + node_b_position + node_c_position) / 3.0;
}




__host__ __device__ void calculateParticleFaceNormal(ParticleFace* faces,
													 ParticleNode* nodes,
													 unsigned int  face_index)
{
	// https://math.stackexchange.com/questions/3698021/how-to-find-if-a-3d-point-is-in-on-outside-of-tetrahedron

	// Get the node positions
	double3 node_a_position = nodes[faces[face_index].nodes[0]].position;
	double3 node_b_position = nodes[faces[face_index].nodes[1]].position;
	double3 node_c_position = nodes[faces[face_index].nodes[2]].position;
	double3 node_d_position = nodes[faces[face_index].nodes[3]].position;

	// Calculate vectors from node A to node B and node C
	double3 vector_a_b = node_b_position - node_a_position;
	double3 vector_a_c = node_c_position - node_a_position;

	// Calculate the unit normal vector
	double3 normal = get_normalize(cross(vector_a_b, vector_a_c));

	// Calculate a unit vector from node D to the barycenter of the face
	double3 vector_d_center = get_normalize(faces[face_index].barycenter - node_d_position);

	// Get the direction of the normal vector to the position of node D
	double direction = dot(normal, vector_d_center);

	// If the direction is negative
	if (direction < 0.0)
	{
		// Then we flip the normal to point outside of the particle
		normal = -normal;
	}

	// Assign the normal to the face
	faces[face_index].normal = normal;

	// Calculate the distance
	faces[face_index].distance = dot(normal, node_a_position);
}


__host__ __device__ void calculateParticleFaceArea(ParticleFace* faces,
												   ParticleNode* nodes,
												   unsigned int  face_index)
{
	// Get the node positions
	double3 node_a_position = nodes[faces[face_index].nodes[0]].position;
	double3 node_b_position = nodes[faces[face_index].nodes[1]].position;
	double3 node_c_position = nodes[faces[face_index].nodes[2]].position;

	// Calculate vectors from node A to node B and node C
	double3 vector_a_b = node_b_position - node_a_position;
	double3 vector_a_c = node_c_position - node_a_position;

	// Calculate the area
	faces[face_index].area = 0.5 * length(cross(vector_a_b, vector_a_c));
}


__host__ __device__ bool pointOnParticleFace(ParticleFace*	faces,
											 ParticleNode*	nodes,
											 double*		coefficients,
											 unsigned int	face_index,
											 double3		point)
{
	// Get the node positions
	double3 node_a_position = nodes[faces[face_index].nodes[0]].position;
	double3 node_b_position = nodes[faces[face_index].nodes[1]].position;
	double3 node_c_position = nodes[faces[face_index].nodes[2]].position;

	// calculate vectors from the point to the face nodes
	double3 vector_point_a = node_a_position - point;
	double3 vector_point_b = node_b_position - point;
	double3 vector_point_c = node_c_position - point;

	// Calculate the area of sub-triangles
	double area_1 = 0.5 * length(cross(vector_point_a, vector_point_b));
	double area_2 = 0.5 * length(cross(vector_point_a, vector_point_c));  
	double area_3 = 0.5 * length(cross(vector_point_b, vector_point_c));

	// Calculate the sum of the sub-areas
	double total_sub_area = area_1 + area_2 + area_3;

	// Check if the total area of the sub triangles equal to the area of the face
	if (areEqual(faces[face_index].area, total_sub_area))
	{
		coefficients[0] = area_3 / faces[face_index].area;
		coefficients[1] = area_2 / faces[face_index].area;
		coefficients[2] = area_1 / faces[face_index].area;
		return true;
	}
	else
	{
		return false;
	}
}



__host__ __device__ void findAdjacentFaceNodes(ParticleFace*	faces,
												ParticleNode*	nodes,
												unsigned int	face_a_index,
												unsigned int	face_b_index,
												int*			number_of_adjacent_nodes,
												int*			face_a_nodes,
												int*			face_b_nodes)
{
	// Go through the nodes of face A
	for (int i = 0; i < 3; i++)
	{
		// Get the index of the node
		int face_a_node_index = faces[face_a_index].nodes[i];

		// Get the position of the node
		double3 face_a_node_position = nodes[face_a_node_index].position;

		// Compare to the nodes of face B
		for (int j = 0; j < 3; j++)
		{
			// Get the index of the node
			int face_b_node_index = faces[face_b_index].nodes[i];

			// Get the position of the node
			double3 face_b_node_position = nodes[face_b_node_index].position;

			// Check if the positions are equal
			if (face_a_node_position == face_b_node_position)
			{
				// Register the nodes
				face_a_nodes[*number_of_adjacent_nodes] = face_a_node_index;
				face_b_nodes[*number_of_adjacent_nodes] = face_b_node_index;

				// Increase the number of adjacent nodes
				number_of_adjacent_nodes++;

				// Break the loop since there are no more matching nodes... theoretically
				break;
			}
		}
	}
}


__host__ void updateParticleFacesCPU(ParticleFace*	faces,
									 unsigned int	number_of_faces,
									 ParticleNode*	nodes,
									 unsigned int	number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](ParticleFace*  faces,
					 unsigned int	number_of_faces,
					 ParticleNode*  nodes,
					 const int		number_of_threads,
					 const int		thread_id)
	{
		int face_index = thread_id;
		while (face_index < number_of_faces)
		{
			calculateParticleFaceCenter(faces, nodes, face_index);
			calculateParticleFaceNormal(faces, nodes, face_index);
			calculateParticleFaceArea(faces, nodes, face_index);
			face_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  faces,
									  number_of_faces,
									  nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}