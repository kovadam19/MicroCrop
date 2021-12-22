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



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____   _               _      ____   _                              _       _   _               _            _____             _                              _     _                 
//   |  ___| (_)  _ __     __| |    / ___| | |   ___    ___    ___   ___  | |_    | \ | |   ___     __| |   ___    |_   _|   ___     | |       ___     ___    __ _  | |_  (_)   ___    _ __  
//   | |_    | | | '_ \   / _` |   | |     | |  / _ \  / __|  / _ \ / __| | __|   |  \| |  / _ \   / _` |  / _ \     | |    / _ \    | |      / _ \   / __|  / _` | | __| | |  / _ \  | '_ \ 
//   |  _|   | | | | | | | (_| |   | |___  | | | (_) | \__ \ |  __/ \__ \ | |_    | |\  | | (_) | | (_| | |  __/     | |   | (_) |   | |___  | (_) | | (__  | (_| | | |_  | | | (_) | | | | |
//   |_|     |_| |_| |_|  \__,_|    \____| |_|  \___/  |___/  \___| |___/  \__|   |_| \_|  \___/   \__,_|  \___|     |_|    \___/    |_____|  \___/   \___|  \__,_|  \__| |_|  \___/  |_| |_|
//                                                                                                                                                                                           
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Finds the closes node to the a given location and returns its index
inline __host__ int findClosestNodeToLocation(Node*			nodes,
											  const int		number_of_nodes,
											  const double3	location)
{
	// Create a variable with infinite distance
	double min_distance = INF;

	// Create a variable for the node index
	int node_index = -1;

	// Go through the nodes and find the closest one to the given location
	for (int j = 0; j < number_of_nodes; j++)
	{
		// Calculate the distance between the node and the given location
		double distance = length(location - nodes[j].position);

		// Check if the current node is closer to the location
		if (distance < min_distance)
		{
			// Update the minimum distance and the node index
			min_distance = distance;
			node_index = j;
		}
	}

	// Return the node index
	return node_index;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____   _               _     _   _               _            ___             _                              _     _                 
//   |  ___| (_)  _ __     __| |   | \ | |   ___     __| |   ___    |_ _|  _ __     | |       ___     ___    __ _  | |_  (_)   ___    _ __  
//   | |_    | | | '_ \   / _` |   |  \| |  / _ \   / _` |  / _ \    | |  | '_ \    | |      / _ \   / __|  / _` | | __| | |  / _ \  | '_ \ 
//   |  _|   | | | | | | | (_| |   | |\  | | (_) | | (_| | |  __/    | |  | | | |   | |___  | (_) | | (__  | (_| | | |_  | | | (_) | | | | |
//   |_|     |_| |_| |_|  \__,_|   |_| \_|  \___/   \__,_|  \___|   |___| |_| |_|   |_____|  \___/   \___|  \__,_|  \__| |_|  \___/  |_| |_|
//                                                                                                                                          
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Finds the node that is exactly located in the given location and returns its index
inline __host__ int findNodeInLocation(Node*			nodes,
									   const int		number_of_nodes,
									   double3	location)
{
	// Create a variable for the node index
	int node_index = -1;

	// Go through the nodes and find the node that is exactly in the location
	for (int j = 0; j < number_of_nodes; j++)
	{
		// Check if the current node is in the given location
		if (nodes[j].position == location)
		{
			// Set the node index
			node_index = j;

			// Break the loop
			break;
		}
	}

	// Return the node index
	return node_index;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____   _               _     ___           _                                  _     _                     ____                                         _           
//   |  ___| (_)  _ __     __| |   |_ _|  _ __   | |_    ___   _ __    __ _    ___  | |_  (_)   ___    _ __     |  _ \   _ __    ___    _ __     ___   _ __  | |_   _   _ 
//   | |_    | | | '_ \   / _` |    | |  | '_ \  | __|  / _ \ | '__|  / _` |  / __| | __| | |  / _ \  | '_ \    | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | | | |
//   |  _|   | | | | | | | (_| |    | |  | | | | | |_  |  __/ | |    | (_| | | (__  | |_  | | | (_) | | | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | |_| |
//   |_|     |_| |_| |_|  \__,_|   |___| |_| |_|  \__|  \___| |_|     \__,_|  \___|  \__| |_|  \___/  |_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__|  \__, |
//                                                                                                                                     |_|                          |___/ 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Finds the interaction properties between two components and returns the index of the interaction property
inline __host__ __device__ int findInteractionProperty(InteractionProperty* interaction_properties,
													   const int			number_of_interactions,
													   int					component_a,
													   int					component_b)
{
	// Check if component A is greater than component B
	if (component_a > component_b)
	{
		// Swap the components
		int temp = component_a;
		component_a = component_b;
		component_b = temp;
	}

	// Go through the interaction properties
	for (int i = 0; i < number_of_interactions; i++)
	{
		// Check if component A is in the interaction property
		if (interaction_properties[i].component_a == component_a)
		{
			// Check if component B is in the interaction property
			if (interaction_properties[i].component_b == component_b)
			{
				return i;
			}
		}
	}

	return -1;
}
