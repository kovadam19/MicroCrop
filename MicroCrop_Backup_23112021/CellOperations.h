#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thread>

#include "Primitives.h"


inline __host__ void calculateNodeMass(Cell* cells,
	Node* nodes,
	int	cell_index)
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

inline __host__ __device__ void calculateCellCenter(Cell*			cells,
											 Node*			nodes,
											 int	cell_index)
{
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



inline __host__ void assignCellMaterial(Cell*				cells,
								 int		cell_index,
								 MaterialProperty*	materials,
								 int		size_of_materials)
{
	// Create a variable with infinite distance
	double min_distance = INF;

	// Create a variable for the material index
	int material_index = -1;

	// Go through the materials and find the closest one to the barycenter of the cell
	for (int j = 0; j < size_of_materials; j++)
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

	// Assign the closest material to the cell
	cells[cell_index].material_type = materials[material_index].type;
	cells[cell_index].material_property = material_index;
	
}



inline __host__ __device__ void calculateCellVolume(Cell*			cells,
											 Node*			nodes,
											 int	cell_index)
{
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


inline __host__ void calculateCellMass(Cell*				cells,
								MaterialProperty*	materials,
								int		cell_index)
{
	// Get the density
	double density = materials[cells[cell_index].material_property].density;

	// Calculate the mass
	cells[cell_index].mass = cells[cell_index].volume * density;
}




inline __host__ __device__ void calculateCellCircumsphere(Cell*			cells,
												   Node*			nodes, 
												   int		cell_index)
{
	// Reset the circumsphere radius
	cells[cell_index].circumsphere_radius = 0.0;

	// Go through the cell nodes
	for (int j = 0; j < 4; j++)
	{
		// Position of the current node
		double3 node_position = nodes[cells[cell_index].nodes[j]].position;
			
		// Distance from the barycenter to the current node
		double distance = length(node_position - cells[cell_index].barycenter);

		// Check if the distance is larger the current circumsphere radius
		if (distance > cells[cell_index].circumsphere_radius)
		{
			// Assign it to the cell
			cells[cell_index].circumsphere_radius = distance;
		}
	}
}



inline __host__ __device__ void checkCellDamage(Cell*				cells,
										 int				cell_index,
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
		double strength = axial_springs[spring_index].strength;

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


inline __host__ void initializeCells(Cell*				cells,
							  int		number_of_cells,
							  Node*				nodes,
							  MaterialProperty* materials,
							  int		size_of_materials,
							  Settings&			settings)
{

	std::vector<std::thread> threads;

	auto initialize = [](Cell*				cells,
						 const int			number_of_cells,
						 Node*				nodes,
						 MaterialProperty*	materials,
						 int		size_of_materials,
						 const int			number_of_threads,
						 const int			thread_id)
	{
		int cell_index = thread_id;
		while (cell_index < number_of_cells)
		{
			calculateCellCenter(cells, nodes, cell_index);
			assignCellMaterial(cells, cell_index, materials, size_of_materials);
			calculateCellVolume(cells, nodes, cell_index);
			calculateCellMass(cells, materials, cell_index);
			calculateCellCircumsphere(cells, nodes, cell_index);
			calculateNodeMass(cells, nodes, cell_index);
			cell_index += number_of_threads;
		}
	};

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

	for (auto& thread : threads)
	{
		thread.join();
	}
}


inline __host__ void updateCellsCPU(Cell* cells,
							int	number_of_cells,
							Node* nodes,
							IntersectionNode* intersection_nodes,
							Face* faces,
							AxialSpring* axial_springs,
							RotationalSpring* rotational_springs,
							int	number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](Cell* cells,
					int	number_of_cells,
					Node* nodes,
					IntersectionNode* intersection_nodes,
					Face* faces,
					AxialSpring* axial_springs,
					RotationalSpring* rotational_springs,
					 const int		number_of_threads,
					 const int		thread_id)
	{
		int cell_index = thread_id;
		while (cell_index < number_of_cells)
		{
			if (cells[cell_index].status == 1)
			{
				calculateCellCenter(cells, nodes, cell_index);
				calculateCellVolume(cells, nodes, cell_index);
				calculateCellCircumsphere(cells, nodes, cell_index);
				checkCellDamage(cells, cell_index, faces, intersection_nodes, axial_springs, rotational_springs);
			}

			cell_index += number_of_threads;
		}
	};

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

	for (auto& thread : threads)
	{
		thread.join();
	}
}