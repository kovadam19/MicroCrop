/****************************************************************
* Project: MicroCrop - Advanced Anisotropic Mass-Spring System
* Author : Adam Kovacs
* Version : 1.0.0
* Maintainer : Adam Kovacs
* Email: kovadam19@gmail.com
* Released: 01 January 2022
*****************************************************************/

#pragma once

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Other includes
#include "Primitives.h"
#include "FaceOperations.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____                         _              ___           _                                       _     _                     _   _               _              
//    / ___|  _ __    ___    __ _  | |_    ___    |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___   ___ 
//   | |     | '__|  / _ \  / _` | | __|  / _ \    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \ / __|
//   | |___  | |    |  __/ | (_| | | |_  |  __/    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/ \__ \
//    \____| |_|     \___|  \__,_|  \__|  \___|   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___| |___/
//                                                                                                                                                                      
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Creates intersection nodes for the axial springs within the cells
inline __host__ void createIntersectionNodes(IntersectionNodeContainer&		intersection_nodes,
											  NodeContainer&				nodes,
											  FaceContainer&				faces,
											  CellContainer&				cells,
											  MaterialContainer&			materials)
{
	// Go through all the cells
	for (int i = 0; i < cells.size(); i++)
	{
		// Get the cell material index
		int material_index = cells[i].material_property;

		// Number of intersection points per cell
		int nodes_per_cell = 0;

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
				int face_index = cells[i].faces[k];

				// Get the position of one of the nodes on the face
				double3 node_a_position = nodes[faces[face_index].nodes[0]].position;

				// Calculate a vector from the cell barycenter to one of the nodes on the face
				double3 center_to_node = node_a_position - cells[i].barycenter;

				// Get the face normal
				double3 face_normal = faces[face_index].normal;

				// Calculate the normal component to the face normal
				double vector_normal_component = dot(center_to_node, face_normal);

				// Calculate the normal component of the axis to the face normal
				double axis_normal_component = dot(axis, face_normal);

				// Calculate the distance from the barycenter to the intersection point
				double distance = vector_normal_component / axis_normal_component;

				// Calculate the intersection point position
				double3 intersection_point = cells[i].barycenter + axis * distance;

				// Check if the point falls within the triangle
				double coefficients[3] = { 0.0, 0.0, 0.0 };

				bool point_on_face = pointOnFace(&faces[0],
												 &nodes[0],
												 coefficients,
												 cells[i].faces[k],
												 intersection_point);

				if (point_on_face)
				{
					// Create a new intersection node
					IntersectionNode new_node;

					// Assign an ID to the intersection node
					new_node.id = _intersection_node_id++;

					// Assign the component to the intersection node
					new_node.component = cells[i].component;

					// Set the status to be active
					new_node.status = 1;

					// Assign the cell to the node
					new_node.cell = i;

					// Assign the nodes to the intersection node
					new_node.nodes[0] = faces[face_index].nodes[0];
					new_node.nodes[1] = faces[face_index].nodes[1];
					new_node.nodes[2] = faces[face_index].nodes[2];

					// Assign the coefficients for the shape function
					new_node.coefficients[0] = coefficients[0];
					new_node.coefficients[1] = coefficients[1];
					new_node.coefficients[2] = coefficients[2];

					// Get the independent node positions
					double3 node_a_position = nodes[new_node.nodes[0]].position;
					double3 node_b_position = nodes[new_node.nodes[1]].position;
					double3 node_c_position = nodes[new_node.nodes[2]].position;

					// Calculate the position of the the intersection node
					new_node.position = new_node.coefficients[0] * node_a_position +
										new_node.coefficients[1] * node_b_position +
										new_node.coefficients[2] * node_c_position;

					// Add the new node to the intersection node container
					intersection_nodes.push_back(new_node);

					// Add the intersection node to the cell
					cells[i].intersections[nodes_per_cell] = intersection_nodes.size() - 1;

					// Increase the number of intersection nodes
					nodes_per_axis++;
					nodes_per_cell++;
				}

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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _              ___           _                                       _     _                     _   _               _            ____                  _   _     _                 
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___    |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___    |  _ \    ___    ___  (_) | |_  (_)   ___    _ __  
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \   | |_) |  / _ \  / __| | | | __| | |  / _ \  | '_ \ 
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/   |  __/  | (_) | \__ \ | | | |_  | | | (_) | | | | |
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___|   |_|      \___/  |___/ |_|  \__| |_|  \___/  |_| |_|
//                                                                                                                                                                                                                                       
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the position of the given intersection node
inline __host__ __device__ void calculateIntersectionNodePosition(IntersectionNode*		intersection_nodes,
																   Node*				nodes,
																   const int			inode_index)
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _              ___           _                                       _     _                     _   _               _           __     __         _                  _   _           
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___    |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___    \ \   / /   ___  | |   ___     ___  (_) | |_   _   _ 
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \    \ \ / /   / _ \ | |  / _ \   / __| | | | __| | | | |
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/     \ V /   |  __/ | | | (_) | | (__  | | | |_  | |_| |
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___|      \_/     \___| |_|  \___/   \___| |_|  \__|  \__, |
//                                                                                                                                                                                                                                   |___/ 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the velocity of the given intersection node
inline __host__ __device__ void calculateIntersectionNodeVelocity(IntersectionNode*	intersection_nodes,
																  Node*				nodes,
																  const int			inode_index)
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _              ___           _                                       _     _                     _   _               _                   ____   ____    _   _ 
//   | | | |  _ __     __| |   __ _  | |_    ___    |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___   ___     / ___| |  _ \  | | | |
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \ / __|   | |     | |_) | | | | |
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/ \__ \   | |___  |  __/  | |_| |
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___| |___/    \____| |_|      \___/ 
//           |_|                                                                                                                                                                                    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the intersection nodes on CPU
inline __host__ void updateIntersectionNodesCPU(IntersectionNode*	intersection_nodes,
												const int			number_of_inodes,
												Node*				nodes,
												const int			number_of_threads)
{
	// Creating threads
	std::vector<std::thread> threads;

	// Creating a lambda function for the update
	auto update = [](IntersectionNode*	intersection_nodes,
					 const int			number_of_inodes,
					 Node*				nodes,
					 const int			number_of_threads,
					 const int			thread_id)
	{
		// Loop through the nodes assigned to the thread
		for (int inode_index = thread_id; inode_index < number_of_inodes; inode_index += number_of_threads)
		{
			// Check if the node is active
			if (intersection_nodes[inode_index].status == 1)
			{
				calculateIntersectionNodePosition(intersection_nodes, nodes, inode_index);
				calculateIntersectionNodeVelocity(intersection_nodes, nodes, inode_index);
			}
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  intersection_nodes,
									  number_of_inodes,
									  nodes,
									  number_of_threads,
									  thread_id));
	}

	// Joining threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}
