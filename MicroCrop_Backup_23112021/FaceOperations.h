#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thread>

#include "Primitives.h"




inline __host__ void createFaces(FaceContainer&	faces,
						  CellContainer&	cells, 
						  NodeContainer&	nodes)
{
	// The first three points define the face (a-b-c) and the fourth one (d) helps to define the normal direction
	int faces_definition[4][4] = { {0, 1, 2, 3},
								   {0, 1, 3, 2},
								   {0, 2, 3, 1},
								   {1, 2, 3, 0} };

	// Loop through the cells
	for (int i = 0; i < cells.size(); i++)
	{
		// Loop through the faces
		for (int j = 0; j < 4; j++)
		{
			// Create a face
			Face new_face;

			// Assign an ID to the new face
			new_face.id = _face_id++;

			// Set the face status to be active
			new_face.status = 1;

			// Assign the cell to the face
			new_face.cell = i;

			// Assign nodes to the face
			for (int k = 0; k < 4; k++)
			{
				new_face.nodes[k] = cells[i].nodes[faces_definition[j][k]];
			}

			// Add the new face to the container
			faces.push_back(new_face);

			// Assign the face to the cell
			cells[i].faces[j] = faces.size() - 1;
		}
	}
}




inline __host__ __device__ void calculateFaceCenter(Face*			faces,
											 Node*			nodes,
											 int	face_index)
{
	// Get the node positions
	double3 node_a_position = nodes[faces[face_index].nodes[0]].position;
	double3 node_b_position = nodes[faces[face_index].nodes[1]].position;
	double3 node_c_position = nodes[faces[face_index].nodes[2]].position;

	// Calculate the barycenter
	faces[face_index].barycenter = (node_a_position + node_b_position + node_c_position) / 3.0;
}




inline __host__ __device__ void calculateFaceNormal(Face*			faces,
											 Node*			nodes,
											 int	face_index)
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
		// Then we flip the normal to point outside of the cell
		normal = -normal;
	}

	// Assign the normal to the face
	faces[face_index].normal = normal;

	// Calculate the distance
	faces[face_index].distance = dot(normal, node_a_position);
}


inline __host__ __device__ void calculateFaceArea(Face*		 faces,
										   Node*		 nodes,
										   int  face_index)
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


inline __host__ __device__ bool pointOnFace(Face*			faces,
									 Node*			nodes,
									 double*		coefficients,
									 int	face_index,
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



inline __host__ void updateFacesCPU(Face*	faces,
									 int	number_of_faces,
									 Node*	nodes,
									 int	number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](Face*  faces,
					 int	number_of_faces,
					 Node*  nodes,
					 const int		number_of_threads,
					 const int		thread_id)
	{
		int face_index = thread_id;
		while (face_index < number_of_faces)
		{
			if (faces[face_index].status == 1)
			{
				calculateFaceCenter(faces, nodes, face_index);
				calculateFaceNormal(faces, nodes, face_index);
				calculateFaceArea(faces, nodes, face_index);
			}
			
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