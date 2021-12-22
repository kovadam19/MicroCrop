#pragma once

#include <thread>



#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "Primitives.h"



inline __host__ void createAxialSprings(AxialSpringContainer&		axial_springs,
								 CellContainer&				cells, 
								 IntersectionNodeContainer&	intersection_nodes, 
								 MaterialContainer&			materials	)
{
	// Go through the cells
	for (int i = 0; i < cells.size(); i++)
	{
		// Get the cell material index
		int material_index = cells[i].material_property;

		// Go through the anisotropy axes (intersection node pairs)
		for (int j = 0; j < 3; j++)
		{
			// Get the intesection node indices
			int node_a_index = cells[i].intersections[2 * j];
			int node_b_index = cells[i].intersections[2 * j + 1];

			// Get the intersection node positions
			double3 node_a_position = intersection_nodes[node_a_index].position;
			double3 node_b_position = intersection_nodes[node_b_index].position;

			// Create a new axial spring
			AxialSpring new_spring;

			// Assign an ID to the new spring
			new_spring.id = _axial_spring_id++;

			// Set the status to be active
			new_spring.status = 1;

			// Assign the cell to the spring
			new_spring.cell = i;

			// Assign the nodes to the new spring
			new_spring.nodes[0] = node_a_index;
			new_spring.nodes[1] = node_b_index;

			// Assign the stiffness to the new spring
			new_spring.stiffness = materials[material_index].axial_stiffnesses[j];

			// Assign the damping to the new spring
			new_spring.damping = materials[material_index].axial_dampings[j];

			// Assign the strength to the new spring
			new_spring.strength = materials[material_index].strength[j];

			// Claculate the initial length of the new spring
			new_spring.initial_length = length(node_b_position - node_a_position);

			// Set the current length to be equal to the initial length
			new_spring.current_length = new_spring.initial_length;

			// Add the new spring to the container
			axial_springs.push_back(new_spring);

			// Assign the spring to the cell
			cells[i].axial_springs[j] = axial_springs.size() - 1;
		}
	}
}



inline __host__ __device__ void calculateAxialSpring(AxialSpring*		axial_springs,
											  IntersectionNode* intersection_nodes,
											  int		spring_index)
{

	//    ___          _   _     _          _      ___          _               _          _     _                   
	//   |_ _|  _ _   (_) | |_  (_)  __ _  | |    / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//    | |  | ' \  | | |  _| | | / _` | | |   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//   |___| |_||_| |_|  \__| |_| \__,_| |_|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//                                                                                                               

	// Get the node indicies
	int node_a_index = axial_springs[spring_index].nodes[0];
	int node_b_index = axial_springs[spring_index].nodes[1];
	
	// Get the node positions
	double3 node_a_position = intersection_nodes[node_a_index].position;
	double3 node_b_position = intersection_nodes[node_b_index].position;
	
	// Claculate the distance vector from node A to node B
	double3 distance_vector = node_b_position - node_a_position;

	// Calculate the unit vector from node A to node B
	double3 unit_vector = get_normalize(distance_vector);

	// Calculte the current length
	double current_length = length(distance_vector);

	// Save the current length
	axial_springs[spring_index].current_length = current_length;

	// Get the initial length
	double initial_length = axial_springs[spring_index].initial_length;

	// Calculate the delta length
	double delta_length = current_length - initial_length;

	//    ___                _                   ___                             ___          _               _          _     _                   
	//   / __|  _ __   _ _  (_)  _ _    __ _    | __|  ___   _ _   __   ___     / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//   \__ \ | '_ \ | '_| | | | ' \  / _` |   | _|  / _ \ | '_| / _| / -_)   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//   |___/ | .__/ |_|   |_| |_||_| \__, |   |_|   \___/ |_|   \__| \___|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//         |_|                     |___/                                                                                                       

	// Get the spring stiffness
	double stiffness = axial_springs[spring_index].stiffness;

	// Calculate the magnitude force
	double force_magnitude = stiffness * fabs(delta_length);

	// Spring force vector
	double3 spring_force_vector = unit_vector * force_magnitude;

	// Check if the delta length is smaller then zero
	if (delta_length < 0.0)
	{
		// Compression
		// Spring force on node_a (since the unit vector points to node_b and we are in compression we have to multiply by -1 the node force on node_a)
		axial_springs[spring_index].spring_force_node_a = -spring_force_vector;

		// Spring force on node_b
		axial_springs[spring_index].spring_force_node_b = spring_force_vector;
	}
	else if (delta_length > 0.0)
	{
		// Tension
		// Spring force on node_a
		axial_springs[spring_index].spring_force_node_a = spring_force_vector;

		// Spring force on node_b (since the unit vector points to node_b and we are in tension we have to multiply by -1 the node force on node_b)
		axial_springs[spring_index].spring_force_node_b = -spring_force_vector;
	}

	//    ___                          _                   ___                             ___          _               _          _     _                   
	//   |   \   __ _   _ __    _ __  (_)  _ _    __ _    | __|  ___   _ _   __   ___     / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//   | |) | / _` | | '  \  | '_ \ | | | ' \  / _` |   | _|  / _ \ | '_| / _| / -_)   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//   |___/  \__,_| |_|_|_| | .__/ |_| |_||_| \__, |   |_|   \___/ |_|   \__| \___|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//                         |_|               |___/                                                                                                       

	// Get the damping
	double damping = axial_springs[spring_index].damping;

	// Get the node velocities
	double3 node_a_velocity = intersection_nodes[node_a_index].velocity;
	double3 node_b_velocity = intersection_nodes[node_b_index].velocity;

	// Calculate the relative velocity of node_a and node_b
	double3 relative_velocity = node_b_velocity - node_a_velocity;

	// Calculate the normal relative velocity of node_a and node_b
	double3 normal_relative_velocity = unit_vector * dot(unit_vector, relative_velocity);

	// Calculate the normal damping force magnitude
	double damping_force_magnitude = damping * length(normal_relative_velocity);

	// Check the direction of the relative normal velocity
	if (dot(normal_relative_velocity, unit_vector) < 0.0)
	{
		// In this case the nodes are getting closer to each other (loading)
		axial_springs[spring_index].damping_force_node_a = -1 * unit_vector * damping_force_magnitude;
		axial_springs[spring_index].damping_force_node_b = unit_vector * damping_force_magnitude;
	}
	else
	{
		// In this case the nodes are getting further from each other (unloading)
		axial_springs[spring_index].damping_force_node_a = unit_vector * damping_force_magnitude;
		axial_springs[spring_index].damping_force_node_b = -1 * unit_vector * damping_force_magnitude;
	}

	//    _____         _            _     ___                             ___          _               _          _     _                   
	//   |_   _|  ___  | |_   __ _  | |   | __|  ___   _ _   __   ___     / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//     | |   / _ \ |  _| / _` | | |   | _|  / _ \ | '_| / _| / -_)   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//     |_|   \___/  \__| \__,_| |_|   |_|   \___/ |_|   \__| \___|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//                                                                                                                                       

	// Calculte the total force on node_a and node_b
	axial_springs[spring_index].total_force_node_a = axial_springs[spring_index].spring_force_node_a + axial_springs[spring_index].damping_force_node_a;
	axial_springs[spring_index].total_force_node_b = axial_springs[spring_index].spring_force_node_b + axial_springs[spring_index].damping_force_node_b;
}






inline __host__ void applyAxialSpringForces(AxialSpring*		axial_springs,
												int		number_of_springs,
												IntersectionNode*	intersection_nodes,
												Node*		nodes)
{
	// Go through the springs
	for (int i = 0; i < number_of_springs; i++)
	{
		// Check if the spring is active
		if (axial_springs[i].status == 1)
		{
			// Get the intersection node indicies
			int inode_a_index = axial_springs[i].nodes[0];
			int inode_b_index = axial_springs[i].nodes[1];

			// Get the spring forces on the intersection nodes
			double3 iforce_a = axial_springs[i].total_force_node_a;
			double3 iforce_b = axial_springs[i].total_force_node_b;

			// Get the cell node indices
			int pnode_a_index = intersection_nodes[inode_a_index].nodes[0];
			int pnode_b_index = intersection_nodes[inode_a_index].nodes[1];
			int pnode_c_index = intersection_nodes[inode_a_index].nodes[2];
			int pnode_d_index = intersection_nodes[inode_b_index].nodes[0];
			int pnode_e_index = intersection_nodes[inode_b_index].nodes[1];
			int pnode_f_index = intersection_nodes[inode_b_index].nodes[2];

			// Get the cell node coefficients
			double pnode_a_coefficient = intersection_nodes[inode_a_index].coefficients[0];
			double pnode_b_coefficient = intersection_nodes[inode_a_index].coefficients[1];
			double pnode_c_coefficient = intersection_nodes[inode_a_index].coefficients[2];
			double pnode_d_coefficient = intersection_nodes[inode_b_index].coefficients[0];
			double pnode_e_coefficient = intersection_nodes[inode_b_index].coefficients[1];
			double pnode_f_coefficient = intersection_nodes[inode_b_index].coefficients[2];

			// Apply the forces on the cell nodes
			nodes[pnode_a_index].force += iforce_a * pnode_a_coefficient;
			nodes[pnode_b_index].force += iforce_a * pnode_b_coefficient;
			nodes[pnode_c_index].force += iforce_a * pnode_c_coefficient;
			nodes[pnode_d_index].force += iforce_b * pnode_d_coefficient;
			nodes[pnode_e_index].force += iforce_b * pnode_e_coefficient;
			nodes[pnode_f_index].force += iforce_b * pnode_f_coefficient;
		}
	}
}





inline __host__ void updateAxialSpringsCPU(AxialSpring*		axial_springs,
									int		number_of_springs,
									IntersectionNode*	intersection_nodes,
									int		number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](AxialSpring*		axial_springs,
					 unsigned int		number_of_springs,
					 IntersectionNode*	intersection_nodes,
					 const int			number_of_threads,
					 const int			thread_id)
	{
		int spring_index = thread_id;
		while (spring_index < number_of_springs)
		{
			if (axial_springs[spring_index].status == 1)
			{
				calculateAxialSpring(axial_springs, intersection_nodes, spring_index);
			}
			
			spring_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  axial_springs,
									  number_of_springs,
									  intersection_nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}



