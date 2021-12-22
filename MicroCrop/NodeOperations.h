/****************************************************************
* Project: MicroCrop - Advanced Anisotropic Mass-Spring System
* Author : Adam Kovacs
* Version : 1.0.0
* Maintainer : Adam Kovacs
* Email: kovadam19@gmail.com
* Released: 01 January 2022
*****************************************************************/

#pragma once

// Generic inludes
#include <thread>

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Other includes
#include "Primitives.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _   _               _             ___              ____    _                        
//   |_ _|  ___    | \ | |   ___     __| |   ___     / _ \   _ __     |  _ \  | |   __ _   _ __     ___ 
//    | |  / __|   |  \| |  / _ \   / _` |  / _ \   | | | | | '_ \    | |_) | | |  / _` | | '_ \   / _ \
//    | |  \__ \   | |\  | | (_) | | (_| | |  __/   | |_| | | | | |   |  __/  | | | (_| | | | | | |  __/
//   |___| |___/   |_| \_|  \___/   \__,_|  \___|    \___/  |_| |_|   |_|     |_|  \__,_| |_| |_|  \___|
//                                                                                                      
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Checks if a node is on a plane (returns 1 or 0)
inline __host__ __device__ int isNodeOnPlane(Node*			nodes,
											 const int		node_index,
											 const double3	location,
											 const double3	normal)
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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _                                   _              _   _               _        
//   |_ _|  _ __   | |_    ___    __ _   _ __    __ _  | |_    ___    | \ | |   ___     __| |   ___ 
//    | |  | '_ \  | __|  / _ \  / _` | | '__|  / _` | | __|  / _ \   |  \| |  / _ \   / _` |  / _ \
//    | |  | | | | | |_  |  __/ | (_| | | |    | (_| | | |_  |  __/   | |\  | | (_) | | (_| | |  __/
//   |___| |_| |_|  \__|  \___|  \__, | |_|     \__,_|  \__|  \___|   |_| \_|  \___/   \__,_|  \___|
//                               |___/                                                              
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Does the time integration on the given node
inline __host__ __device__ void integrateNode(Node*			nodes,
											  const int		node_index,
											  const double	timestep,
											  const double	global_damping)
{
	// Check if there is a fixed velocity assigned to the node
	if (nodes[node_index].apply_fixed_velocity == 1)
	{
		// Calculate the position of the node based on the fixed velocity
		nodes[node_index].position += nodes[node_index].fixed_velocity * timestep;
	}
	else
	{
		// Check the boundary condition in direction X
		if (nodes[node_index].boundaries.x == 0)
		{
			// Time integration
			nodes[node_index].acceleration.x = nodes[node_index].force.x / nodes[node_index].mass;
			nodes[node_index].velocity.x += nodes[node_index].acceleration.x * timestep;
			nodes[node_index].velocity.x -= global_damping * nodes[node_index].velocity.x;
			nodes[node_index].position.x += nodes[node_index].velocity.x * timestep;
		}
		else
		{
			// The movement is fixed
			nodes[node_index].acceleration.x = 0.0;
			nodes[node_index].velocity.x = 0.0;
		}

		// Check the boundary condition in direction Y
		if (nodes[node_index].boundaries.y == 0)
		{
			// Time integration
			nodes[node_index].acceleration.y = nodes[node_index].force.y / nodes[node_index].mass;
			nodes[node_index].velocity.y += nodes[node_index].acceleration.y * timestep;
			nodes[node_index].velocity.y -= global_damping * nodes[node_index].velocity.y;
			nodes[node_index].position.y += nodes[node_index].velocity.y * timestep;
		}
		else
		{
			// The movement is fixed
			nodes[node_index].acceleration.y = 0.0;
			nodes[node_index].velocity.y = 0.0;
		}

		// Check the boundary condition in direction Z
		if (nodes[node_index].boundaries.z == 0)
		{
			// Time integration
			nodes[node_index].acceleration.z = nodes[node_index].force.z / nodes[node_index].mass;
			nodes[node_index].velocity.z += nodes[node_index].acceleration.z * timestep;
			nodes[node_index].velocity.z -= global_damping * nodes[node_index].velocity.z;
			nodes[node_index].position.z += nodes[node_index].velocity.z * timestep;
		}
		else
		{
			// The movement is fixed
			nodes[node_index].acceleration.z = 0.0;
			nodes[node_index].velocity.z = 0.0;
		}
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____                        _       _   _               _                   ____   ____    _   _ 
//   |  _ \    ___   ___    ___  | |_    | \ | |   ___     __| |   ___   ___     / ___| |  _ \  | | | |
//   | |_) |  / _ \ / __|  / _ \ | __|   |  \| |  / _ \   / _` |  / _ \ / __|   | |     | |_) | | | | |
//   |  _ <  |  __/ \__ \ |  __/ | |_    | |\  | | (_) | | (_| | |  __/ \__ \   | |___  |  __/  | |_| |
//   |_| \_\  \___| |___/  \___|  \__|   |_| \_|  \___/   \__,_|  \___| |___/    \____| |_|      \___/ 
//                                                                                                     
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Resets the nodes by using CPU threads
inline __host__ void resetNodesCPU(Node*		nodes,
								   const int	number_of_nodes,
								   const int	number_of_threads)
{
	// Creating container for the threads
	std::vector<std::thread> threads;

	// Creating a lambda function for the reset
	auto reset = [](Node*		nodes,
					const int	number_of_nodes,
					const int	number_of_threads,
					const int	thread_id)
	{
		// Loops through and resets the nodes
		int node_index = thread_id;
		while (node_index < number_of_nodes)
		{
			nodes[node_index].force = make_double3(0.0, 0.0, 0.0);
			node_index += number_of_threads;
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(reset,
									  std::ref(nodes),
									  number_of_nodes,
									  number_of_threads,
									  thread_id));
	}

	// Joining threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _              _   _               _                   ____   ____    _   _ 
//   | | | |  _ __     __| |   __ _  | |_    ___    | \ | |   ___     __| |   ___   ___     / ___| |  _ \  | | | |
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \   |  \| |  / _ \   / _` |  / _ \ / __|   | |     | |_) | | | | |
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/   | |\  | | (_) | | (_| | |  __/ \__ \   | |___  |  __/  | |_| |
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   |_| \_|  \___/   \__,_|  \___| |___/    \____| |_|      \___/ 
//           |_|                                                                                                  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the nodes on CPU threads
inline __host__ void updateNodesCPU(Node*			nodes,
									const int		number_of_nodes,
									const int		number_of_threads,
									const double	timestep,
									const double	global_damping)
{
	// Creating a container for the threads
	std::vector<std::thread> threads;

	// Creating a lambda function for the update
	auto update = [](Node*			nodes,
					 const int		number_of_nodes,
					 const double	timestep,
					 const double	global_damping,
					 const int		number_of_threads,
					 const int		thread_id)
	{
		// Loop the nodes through
		for (int node_index = thread_id; node_index < number_of_nodes; node_index += number_of_threads)
		{
			integrateNode(nodes, node_index, timestep, global_damping);
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  nodes,
									  number_of_nodes,
									  timestep,
									  global_damping,
									  number_of_threads,
									  thread_id));
	}

	// Joining threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}
