#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thread>

#include "Primitives.h"

inline __host__ void createRotationalSprings(RotationalSpringContainer&	rotational_springs,
									  CellContainer&				cells, 
									  AxialSpringContainer&			axial_springs, 
									  IntersectionNodeContainer&	intersection_nodes, 
									  MaterialContainer&			materials)
{
	// Go through the cells
	for (int i = 0; i < cells.size(); i++)
	{
		// Get the cell material index
		int material_index = cells[i].material_property;

		// Go through the axial springs
		for (int j = 0; j < 3; j++)
		{
			// Get the rotational stiffness
			double stiffness = materials[material_index].rotational_stiffnesses[j];

			// Calculate the spring locations
			int spring_a_location;
			int spring_b_location;
			if (j == 2)
			{
				spring_a_location = j;
				spring_b_location = 0;
			}
			else
			{
				spring_a_location = j;
				spring_b_location = j + 1;
			}

			// Get the axial spring indices
			int spring_a_index = cells[i].axial_springs[spring_a_location];
			int spring_b_index = cells[i].axial_springs[spring_b_location];

			// Get the intersection node indices
			int spring_a_node_a_index = axial_springs[spring_a_index].nodes[0];
			int spring_a_node_b_index = axial_springs[spring_a_index].nodes[1];
			int spring_b_node_a_index = axial_springs[spring_b_index].nodes[0];
			int spring_b_node_b_index = axial_springs[spring_b_index].nodes[1];

			// Calculate the normals for spring A and B
			double3 spring_a_normal = get_normalize(intersection_nodes[spring_a_node_a_index].position - intersection_nodes[spring_a_node_b_index].position);
			double3 spring_b_normal = get_normalize(intersection_nodes[spring_b_node_a_index].position - intersection_nodes[spring_b_node_b_index].position);

			// Calculate the initial angle between the springs
			double initial_angle = angle(spring_a_normal, spring_b_normal);

			// Create a new rotationa spring
			RotationalSpring new_spring;

			// Assign an ID to the spring
			new_spring.id = _rotational_spring_id++;

			// Set the status to be active
			new_spring.status = 1;

			// Assign the cell to the spring
			new_spring.cell = i;

			// Assign axial springs to the rotation one
			new_spring.axial_springs[0] = spring_a_index;
			new_spring.axial_springs[1] = spring_b_index;

			// Assign the stiffness to the spring
			new_spring.stiffness = stiffness;

			// Assign the initial angle to the spring
			new_spring.initial_angle = initial_angle;

			// Assigne the current angle to be equal to the initial angle
			new_spring.current_angle = new_spring.initial_angle;

			// Add the new rotational spring to the container
			rotational_springs.push_back(new_spring);

			// Assign the spring to the cell
			cells[i].rotational_springs[j] = rotational_springs.size() - 1;
		}
	}
}


inline __host__ __device__ void calculateRotationalSpring(RotationalSpring*	rotational_springs,
												   AxialSpring*			axial_springs,
												   IntersectionNode*	intersection_nodes,
												   int			rspring_index)
{
	// Get the axial spring indices
	int spring_a_index = rotational_springs[rspring_index].axial_springs[0];
	int spring_b_index = rotational_springs[rspring_index].axial_springs[1];

	// Get the intersection node indices
	int spring_a_node_a_index = axial_springs[spring_a_index].nodes[0];
	int spring_a_node_b_index = axial_springs[spring_a_index].nodes[1];
	int spring_b_node_a_index = axial_springs[spring_b_index].nodes[0];
	int spring_b_node_b_index = axial_springs[spring_b_index].nodes[1];

	// Calculate the normals for spring A and B
	double3 spring_a_normal = get_normalize(intersection_nodes[spring_a_node_a_index].position - intersection_nodes[spring_a_node_b_index].position);
	double3 spring_b_normal = get_normalize(intersection_nodes[spring_b_node_a_index].position - intersection_nodes[spring_b_node_b_index].position);

	// Calculate the current angle between the springs
	double current_angle = angle(spring_a_normal, spring_b_normal);

	// Assign the current angle to the spring
	rotational_springs[rspring_index].current_angle = current_angle;

	// Calculate the delta angle
	double delta_angle = current_angle - rotational_springs[rspring_index].initial_angle;

	// Get the spring stiffness
	double stiffness = rotational_springs[rspring_index].stiffness;

	// Calculate the spring reaction forces
	rotational_springs[rspring_index].spring_a_node_a_force = delta_angle * stiffness * spring_b_normal;
	rotational_springs[rspring_index].spring_a_node_b_force = (-1) * rotational_springs[rspring_index].spring_a_node_a_force;
	rotational_springs[rspring_index].spring_b_node_a_force = delta_angle * stiffness * spring_a_normal;
	rotational_springs[rspring_index].spring_b_node_b_force = (-1) * rotational_springs[rspring_index].spring_b_node_a_force;
}


inline __host__ __device__ void applyRotationalSpringForces(RotationalSpring*	rotational_springs,
													 int		number_of_rsprings,
													 AxialSpring*		axial_springs,
													 IntersectionNode*	intersection_nodes,
													 Node*		nodes)
{
	// Go through the springs
	for (int i = 0; i < number_of_rsprings; i++)
	{
		// Check if the spring is active
		if (rotational_springs[i].status == 1)
		{
			// Get the indices of the axial springs
			int aspring_a_index = rotational_springs[i].axial_springs[0];
			int aspring_b_index = rotational_springs[i].axial_springs[1];

			// Get the intersection node indices
			int aspring_a_inode_a_index = axial_springs[aspring_a_index].nodes[0];
			int aspring_a_inode_b_index = axial_springs[aspring_a_index].nodes[1];
			int aspring_b_inode_a_index = axial_springs[aspring_b_index].nodes[0];
			int aspring_b_inode_b_index = axial_springs[aspring_b_index].nodes[1];

			// Get the cell node indices
			int aspring_a_inode_a_pnode_a_index = intersection_nodes[aspring_a_inode_a_index].nodes[0];
			int aspring_a_inode_a_pnode_b_index = intersection_nodes[aspring_a_inode_a_index].nodes[1];
			int aspring_a_inode_a_pnode_c_index = intersection_nodes[aspring_a_inode_a_index].nodes[2];
			int aspring_a_inode_b_pnode_a_index = intersection_nodes[aspring_a_inode_b_index].nodes[0];
			int aspring_a_inode_b_pnode_b_index = intersection_nodes[aspring_a_inode_b_index].nodes[1];
			int aspring_a_inode_b_pnode_c_index = intersection_nodes[aspring_a_inode_b_index].nodes[2];
			int aspring_b_inode_a_pnode_a_index = intersection_nodes[aspring_b_inode_a_index].nodes[0];
			int aspring_b_inode_a_pnode_b_index = intersection_nodes[aspring_b_inode_a_index].nodes[1];
			int aspring_b_inode_a_pnode_c_index = intersection_nodes[aspring_b_inode_a_index].nodes[2];
			int aspring_b_inode_b_pnode_a_index = intersection_nodes[aspring_b_inode_b_index].nodes[0];
			int aspring_b_inode_b_pnode_b_index = intersection_nodes[aspring_b_inode_b_index].nodes[1];
			int aspring_b_inode_b_pnode_c_index = intersection_nodes[aspring_b_inode_b_index].nodes[2];

			// Get the cell node coefficients
			double aspring_a_inode_a_pnode_a_coefficient = intersection_nodes[aspring_a_inode_a_index].coefficients[0];
			double aspring_a_inode_a_pnode_b_coefficient = intersection_nodes[aspring_a_inode_a_index].coefficients[1];
			double aspring_a_inode_a_pnode_c_coefficient = intersection_nodes[aspring_a_inode_a_index].coefficients[2];
			double aspring_a_inode_b_pnode_a_coefficient = intersection_nodes[aspring_a_inode_b_index].coefficients[0];
			double aspring_a_inode_b_pnode_b_coefficient = intersection_nodes[aspring_a_inode_b_index].coefficients[1];
			double aspring_a_inode_b_pnode_c_coefficient = intersection_nodes[aspring_a_inode_b_index].coefficients[2];
			double aspring_b_inode_a_pnode_a_coefficient = intersection_nodes[aspring_b_inode_a_index].coefficients[0];
			double aspring_b_inode_a_pnode_b_coefficient = intersection_nodes[aspring_b_inode_a_index].coefficients[1];
			double aspring_b_inode_a_pnode_c_coefficient = intersection_nodes[aspring_b_inode_a_index].coefficients[2];
			double aspring_b_inode_b_pnode_a_coefficient = intersection_nodes[aspring_b_inode_b_index].coefficients[0];
			double aspring_b_inode_b_pnode_b_coefficient = intersection_nodes[aspring_b_inode_b_index].coefficients[1];
			double aspring_b_inode_b_pnode_c_coefficient = intersection_nodes[aspring_b_inode_b_index].coefficients[2];

			// Get the rotational spring forces
			double3 aspring_a_inode_a_force = rotational_springs[i].spring_a_node_a_force;
			double3 aspring_a_inode_b_force = rotational_springs[i].spring_a_node_b_force;
			double3 aspring_b_inode_a_force = rotational_springs[i].spring_b_node_a_force;
			double3 aspring_b_inode_b_force = rotational_springs[i].spring_b_node_b_force;

			// Apply the forces
			nodes[aspring_a_inode_a_pnode_a_index].force += aspring_a_inode_a_force * aspring_a_inode_a_pnode_a_coefficient;
			nodes[aspring_a_inode_a_pnode_b_index].force += aspring_a_inode_a_force * aspring_a_inode_a_pnode_b_coefficient;
			nodes[aspring_a_inode_a_pnode_c_index].force += aspring_a_inode_a_force * aspring_a_inode_a_pnode_c_coefficient;
			nodes[aspring_a_inode_b_pnode_a_index].force += aspring_a_inode_b_force * aspring_a_inode_b_pnode_a_coefficient;
			nodes[aspring_a_inode_b_pnode_b_index].force += aspring_a_inode_b_force * aspring_a_inode_b_pnode_b_coefficient;
			nodes[aspring_a_inode_b_pnode_c_index].force += aspring_a_inode_b_force * aspring_a_inode_b_pnode_c_coefficient;
			nodes[aspring_b_inode_a_pnode_a_index].force += aspring_b_inode_a_force * aspring_b_inode_a_pnode_a_coefficient;
			nodes[aspring_b_inode_a_pnode_b_index].force += aspring_b_inode_a_force * aspring_b_inode_a_pnode_b_coefficient;
			nodes[aspring_b_inode_a_pnode_c_index].force += aspring_b_inode_a_force * aspring_b_inode_a_pnode_c_coefficient;
			nodes[aspring_b_inode_b_pnode_a_index].force += aspring_b_inode_b_force * aspring_b_inode_b_pnode_a_coefficient;
			nodes[aspring_b_inode_b_pnode_b_index].force += aspring_b_inode_b_force * aspring_b_inode_b_pnode_b_coefficient;
			nodes[aspring_b_inode_b_pnode_c_index].force += aspring_b_inode_b_force * aspring_b_inode_b_pnode_c_coefficient;
		}
	}
}

inline __host__ void updateRotationalSpringsCPU(RotationalSpring*	rotational_springs,
										 int		number_of_rsprings,
										 AxialSpring*		axial_springs,
										 IntersectionNode*	intersection_nodes,
										 int		number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](RotationalSpring*	rotational_springs,
					 unsigned int		number_of_rsprings,
					 AxialSpring*		axial_springs,
					 IntersectionNode*	intersection_nodes,
					 const int			number_of_threads,
					 const int			thread_id)
	{
		int rspring_index = thread_id;
		while (rspring_index < number_of_rsprings)
		{
			if (rotational_springs[rspring_index].status == 1)
			{
				calculateRotationalSpring(rotational_springs, axial_springs, intersection_nodes, rspring_index);
			}
			
			rspring_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  rotational_springs,
									  number_of_rsprings,
									  axial_springs,
									  intersection_nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}