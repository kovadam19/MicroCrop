/****************************************************************
* Project: MicroCrop - Advanced Anisotropic Mass-Spring System
* Author : Adam Kovacs
* Version : 1.0.0
* Maintainer : Adam Kovacs
* Email: kovadam19@gmail.com
* Released: 01 January 2022
*****************************************************************/

#pragma once

// Generic includes
#include <thread>

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Other includes
#include "Primitives.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _               ____          _   _      ____                  _                 
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___     / ___|   ___  | | | |    / ___|   ___   _ __   | |_    ___   _ __ 
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   | |      / _ \ | | | |   | |      / _ \ | '_ \  | __|  / _ \ | '__|
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |___  |  __/ | | | |   | |___  |  __/ | | | | | |_  |  __/ | |   
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|    \____|  \___| |_| |_|    \____|  \___| |_| |_|  \__|  \___| |_|   
//                                                                                                                                    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the barycenter of the given cell
inline __host__ __device__ void calculateCellCenter(Cell*		cells,
													Node*		nodes,
													const int	cell_index)
{
	// Implemented from
	// https://math.stackexchange.com/questions/1592128/finding-center-of-mass-for-tetrahedron

	// Zero the current barycenter
	cells[cell_index].barycenter = make_double3(0.0, 0.0, 0.0);

	// Go through the cell nodes
	for (int j = 0; j < 4; j++)
	{
		// Add the node positions to the barycenter
		cells[cell_index].barycenter += nodes[cells[cell_index].nodes[j]].position;
	}

	// Divide the coordinates of the barycenter by 4
	cells[cell_index].barycenter *= 0.25;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _                  _                      ____          _   _     __  __           _                   _           _ 
//      / \     ___   ___  (_)   __ _   _ __      / ___|   ___  | | | |   |  \/  |   __ _  | |_    ___   _ __  (_)   __ _  | |
//     / _ \   / __| / __| | |  / _` | | '_ \    | |      / _ \ | | | |   | |\/| |  / _` | | __|  / _ \ | '__| | |  / _` | | |
//    / ___ \  \__ \ \__ \ | | | (_| | | | | |   | |___  |  __/ | | | |   | |  | | | (_| | | |_  |  __/ | |    | | | (_| | | |
//   /_/   \_\ |___/ |___/ |_|  \__, | |_| |_|    \____|  \___| |_| |_|   |_|  |_|  \__,_|  \__|  \___| |_|    |_|  \__,_| |_|
//                              |___/                                                                                         
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Assigns the closest material to the cell
inline __host__ void assignCellMaterial(Cell*				cells,
										const int			cell_index,
										MaterialProperty*	materials,
										const int			size_of_materials)
{
	// Create a variable with infinite distance
	double min_distance = INF;

	// Create a variable for the material index
	int material_index = -1;

	// Go through the materials and find the closest one to the barycenter of the cell
	for (int j = 0; j < size_of_materials; j++)
	{
		// Check if the material property and the cell belong to the same component
		if (cells[cell_index].component == materials[j].component)
		{
			// Calculate the distance between the barycenter of the cell and location of the material property
			double distance = length(materials[j].location - cells[cell_index].barycenter);

			// Check if the current material is closer to the cell the current minimum
			if (distance < min_distance)
			{
				min_distance = distance;
				material_index = j;
			}
		}
	}

	// Assign the closest material to the cell
	cells[cell_index].material_property = material_index;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _               ____          _   _    __     __          _                            
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___     / ___|   ___  | | | |   \ \   / /   ___   | |  _   _   _ __ ___     ___ 
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   | |      / _ \ | | | |    \ \ / /   / _ \  | | | | | | | '_ ` _ \   / _ \
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |___  |  __/ | | | |     \ V /   | (_) | | | | |_| | | | | | | | |  __/
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|    \____|  \___| |_| |_|      \_/     \___/  |_|  \__,_| |_| |_| |_|  \___|
//                                                                                                                                          
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the volume of the given cell
inline __host__ __device__ void calculateCellVolume(Cell*		cells,
													Node*		nodes,
													const int	cell_index)
{
	// Implemented from
	// https://www.vedantu.com/question-answer/find-the-volume-of-tetrahedron-whose-vertices-class-10-maths-cbse-5eedf688ccf1522a9847565b

	// Get the node positions
	// Node A -> 0; Node B -> 1; Node C -> 2; Node D -> 3
	double3 node_a_position = nodes[cells[cell_index].nodes[0]].position;
	double3 node_b_position = nodes[cells[cell_index].nodes[1]].position;
	double3 node_c_position = nodes[cells[cell_index].nodes[2]].position;
	double3 node_d_position = nodes[cells[cell_index].nodes[3]].position;

	// Calculate the edge vectors
	double3 vector_ab = node_b_position - node_a_position;
	double3 vector_ac = node_c_position - node_a_position;
	double3 vector_ad = node_d_position - node_a_position;

	// Calculate the cross product of AB & AC
	double3 cross_product_ab_ac = cross(vector_ab, vector_ac);

	// Calculate the dot product of AD and the cross product of AB & AC
	double volume_parallelepipedon = dot(vector_ad, cross_product_ab_ac);

	// Calculate the volume of the tetrahedron
	cells[cell_index].volume = fabs(volume_parallelepipedon / 6.0);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _               ____          _   _     __  __                     
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___     / ___|   ___  | | | |   |  \/  |   __ _   ___   ___ 
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   | |      / _ \ | | | |   | |\/| |  / _` | / __| / __|
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |___  |  __/ | | | |   | |  | | | (_| | \__ \ \__ \
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|    \____|  \___| |_| |_|   |_|  |_|  \__,_| |___/ |___/
//                                                                                                                      
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the total mass of the cell
inline __host__ void calculateCellMass(Cell*				cells,
									   MaterialProperty*	materials,
									   const int			cell_index)
{
	// Get the density
	double density = materials[cells[cell_index].material_property].density;

	// Calculate the mass
	cells[cell_index].mass = cells[cell_index].volume * density;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _              _   _               _            __  __                     
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___    | \ | |   ___     __| |   ___    |  \/  |   __ _   ___   ___ 
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   |  \| |  / _ \   / _` |  / _ \   | |\/| |  / _` | / __| / __|
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |\  | | (_) | | (_| | |  __/   | |  | | | (_| | \__ \ \__ \
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|   |_| \_|  \___/   \__,_|  \___|   |_|  |_|  \__,_| |___/ |___/
//                                                                                                                              
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the nodal mass of a given cell
inline __host__ void calculateNodeMass(Cell*		cells,
									   Node*		nodes,
									   const int	cell_index)
{
	// Calculate the nodal mass
	double nodal_mass = cells[cell_index].mass / 4.0;

	// Go through the cell nodes
	for (int j = 0; j < 4; j++)
	{
		// Add the mass to the node
		nodes[cells[cell_index].nodes[j]].mass += nodal_mass;
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____            _           _       ___              ____          _   _ 
//   |  _ \    ___   (_)  _ __   | |_    |_ _|  _ __      / ___|   ___  | | | |
//   | |_) |  / _ \  | | | '_ \  | __|    | |  | '_ \    | |      / _ \ | | | |
//   |  __/  | (_) | | | | | | | | |_     | |  | | | |   | |___  |  __/ | | | |
//   |_|      \___/  |_| |_| |_|  \__|   |___| |_| |_|    \____|  \___| |_| |_|
//                                                                             
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Checks if given point falls within a cell
// If true then returns the index of the closest cell face to the point
// Otherwise it returns -1 (invalid face index)
inline __host__ __device__ int pointInCell(Cell*		cells,
										   const int	cell_index,
										   Face*		faces,
										   double3		point)
{
	// Implemented from
	// https://math.stackexchange.com/questions/3698021/how-to-find-if-a-3d-point-is-in-on-outside-of-tetrahedron

	// Go through the particle faces
	double min_distance = 1e25;
	int closest_face_index = -1;
	for (int i = 0; i < 4; i++)
	{
		// Get the face index
		int face_index = cells[cell_index].faces[i];

		// Get the face normal
		double3 face_normal = faces[face_index].normal;

		// Get the face distance
		double face_distance = faces[face_index].distance;

		// Calculate the point distance
		double point_distance = dot(face_normal, point) - face_distance;

		// Check if the point distance is greater than or equal to zero
		if (point_distance > 0.0 || isZero(point_distance))
		{
			return -1;
		}
		else
		{
			// Calculate the magnitude of the distance (overlap)
			double magnitude_distance = fabs(point_distance);

			// Check if the current magnitude is smaller than the previous stored one
			if (magnitude_distance < min_distance)
			{
				// Update the minimum distance and the index of the closest face
				min_distance = magnitude_distance;
				closest_face_index = face_index;
			}
		}
	}

	// If we reach this point then the node is inside the cell and we return the face index closest to the point
	return closest_face_index;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____   _                     _         ____          _   _     ____                                             
//    / ___| | |__     ___    ___  | | __    / ___|   ___  | | | |   |  _ \    __ _   _ __ ___     __ _    __ _    ___ 
//   | |     | '_ \   / _ \  / __| | |/ /   | |      / _ \ | | | |   | | | |  / _` | | '_ ` _ \   / _` |  / _` |  / _ \
//   | |___  | | | | |  __/ | (__  |   <    | |___  |  __/ | | | |   | |_| | | (_| | | | | | | | | (_| | | (_| | |  __/
//    \____| |_| |_|  \___|  \___| |_|\_\    \____|  \___| |_| |_|   |____/   \__,_| |_| |_| |_|  \__,_|  \__, |  \___|
//                                                                                                        |___/        
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Checks if the given cell is damaged
inline __host__ __device__ void checkCellDamage(Cell*				cells,
												const int			cell_index,
												Face*				faces,
												IntersectionNode*	intersection_nodes,
												AxialSpring*		axial_springs,
												RotationalSpring*	rotational_springs)
{
	// Go through the axial springs
	for (int j = 0; j < 3; j++)
	{
		// Get the index of the spring
		int spring_index = cells[cell_index].axial_springs[j];

		// Get the total force
		double3 total_force = axial_springs[spring_index].total_force_node_a;

		// Calculate the magnitude of the total force
		double total_force_magnitude = length(total_force);

		// Get the strength of the spring
		double strength = 1e15;
		if (axial_springs[spring_index].loadcase == 1)
		{
			// The spring is under tension
			strength = axial_springs[spring_index].tensile_strength;
		}
		else if (axial_springs[spring_index].loadcase == 2)
		{
			// The spring is under compression
			strength = axial_springs[spring_index].compressive_strength;
		}
		
		// Check if the magnitude of the total force is larger than the strength of the spring
		if (total_force_magnitude > strength)
		{
			// The cell is damaged

			// Set the cell status to be damaged
			cells[cell_index].status = 2;

			// Set the faces to be damaged
			for (int k = 0; k < 4; k++)
			{
				faces[cells[cell_index].faces[k]].status = 2;
			}

			// Set the intersection nodes to be damaged
			for (int k = 0; k < 6; k++)
			{
				intersection_nodes[cells[cell_index].intersections[k]].status = 2;
			}

			// Set the axial springs to be damaged
			for (int k = 0; k < 3; k++)
			{
				axial_springs[cells[cell_index].axial_springs[k]].status = 2;
			}

			// Set the rotational springs to be damaged
			for (int k = 0; k < 3; k++)
			{
				rotational_springs[cells[cell_index].rotational_springs[k]].status = 2;
			}

			// Break the loop
			break;
		}
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _   _     _           _   _                   ____          _   _       
//   |_ _|  _ __   (_) | |_  (_)   __ _  | | (_)  ____   ___     / ___|   ___  | | | |  ___ 
//    | |  | '_ \  | | | __| | |  / _` | | | | | |_  /  / _ \   | |      / _ \ | | | | / __|
//    | |  | | | | | | | |_  | | | (_| | | | | |  / /  |  __/   | |___  |  __/ | | | | \__ \
//   |___| |_| |_| |_|  \__| |_|  \__,_| |_| |_| /___|  \___|    \____|  \___| |_| |_| |___/
//                                                                                          
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Initializes the cells on multiple CPU threads
inline __host__ void initializeCells(Cell*				cells,
									 const int			number_of_cells,
									 Node*				nodes,
									 MaterialProperty*	materials,
									 const int			size_of_materials,
									 Settings&			settings)
{
	// Creating a container for the threads
	std::vector<std::thread> threads;

	// Creating a lambda function for the initialization
	auto initialize = [](Cell*				cells,
						 const int			number_of_cells,
						 Node*				nodes,
						 MaterialProperty*	materials,
						 int				size_of_materials,
						 const int			number_of_threads,
						 const int			thread_id)
	{
		// Loop through the cells
		for (int cell_index = thread_id; cell_index < number_of_cells; cell_index += number_of_threads)
		{
			calculateCellCenter(cells, nodes, cell_index);
			assignCellMaterial(cells, cell_index, materials, size_of_materials);
			calculateCellVolume(cells, nodes, cell_index);
			calculateCellMass(cells, materials, cell_index);
			calculateNodeMass(cells, nodes, cell_index);
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < settings.number_of_CPU_threads; thread_id++)
	{
		threads.push_back(std::thread(initialize,
									  cells,
									  number_of_cells,
									  nodes,
									  materials,
									  size_of_materials,
									  settings.number_of_CPU_threads,
									  thread_id));
	}

	// Joining the threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _               ____          _   _            ____   ____    _   _ 
//   | | | |  _ __     __| |   __ _  | |_    ___     / ___|   ___  | | | |  ___     / ___| |  _ \  | | | |
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \   | |      / _ \ | | | | / __|   | |     | |_) | | | | |
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/   | |___  |  __/ | | | | \__ \   | |___  |  __/  | |_| |
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|    \____|  \___| |_| |_| |___/    \____| |_|      \___/ 
//           |_|                                                                                          
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the cells on multiple CPU threads
inline __host__ void updateCellsCPU(Cell*				cells,
									const int			number_of_cells,
									Node*				nodes,
									IntersectionNode*	intersection_nodes,
									Face*				faces,
									AxialSpring*		axial_springs,
									RotationalSpring*	rotational_springs,
									const int			number_of_threads)
{
	// Creating a container for the threads
	std::vector<std::thread> threads;

	// Creating an update lambda function
	auto update = [](Cell*				cells,
					 int				number_of_cells,
					 Node*				nodes,
					 IntersectionNode*	intersection_nodes,
					 Face*				faces,
					 AxialSpring*		axial_springs,
					 RotationalSpring*	rotational_springs,
					 const int			number_of_threads,
					 const int			thread_id)
	{
		// Go through the cells
		for (int cell_index = thread_id; cell_index < number_of_cells; cell_index += number_of_threads)
		{
			// Check if the cell is active
			if (cells[cell_index].status == 1)
			{
				calculateCellCenter(cells, nodes, cell_index);
				calculateCellVolume(cells, nodes, cell_index);
				checkCellDamage(cells, cell_index, faces, intersection_nodes, axial_springs, rotational_springs);
			}
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  cells,
									  number_of_cells,
									  nodes,
									  intersection_nodes,
									  faces,
									  axial_springs,
									  rotational_springs,
									  number_of_threads,
									  thread_id));
	}

	// Joining threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}
