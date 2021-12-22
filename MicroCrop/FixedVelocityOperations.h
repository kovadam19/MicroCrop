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
#include <fstream>

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Other includes
#include "Primitives.h"
#include "SearchOperations.h"
#include "NodeOperations.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _                      _             _____   _                     _    __     __         _                  _   _     _              
//      / \     _ __    _ __   | |  _   _    |  ___| (_) __  __   ___    __| |   \ \   / /   ___  | |   ___     ___  (_) | |_  (_)   ___   ___ 
//     / _ \   | '_ \  | '_ \  | | | | | |   | |_    | | \ \/ /  / _ \  / _` |    \ \ / /   / _ \ | |  / _ \   / __| | | | __| | |  / _ \ / __|
//    / ___ \  | |_) | | |_) | | | | |_| |   |  _|   | |  >  <  |  __/ | (_| |     \ V /   |  __/ | | | (_) | | (__  | | | |_  | | |  __/ \__ \
//   /_/   \_\ | .__/  | .__/  |_|  \__, |   |_|     |_| /_/\_\  \___|  \__,_|      \_/     \___| |_|  \___/   \___| |_|  \__| |_|  \___| |___/
//             |_|     |_|          |___/                                                                                                      
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Parses the input files and applies the fixed velocities
inline __host__ void applyFixedVelocities(NodeContainer&	nodes,
										  Settings&			settings)
{
	// Go through the fixed velocity file names
	for (auto& file_name : settings.FixedVelocityFiles)
	{
		// Open the current file
		std::ifstream MyFile(file_name);

		// Create a string for the items
		std::string item;

		// Variable for the number of velocities
		int number_of_velocities = 0;

		// Reads the file until it ends
		while (MyFile >> item)
		{
			// Check if the item is the GLOBAL keyword
			if (item == "GLOBAL")
			{
				// Read in the number of velocities we have to read
				MyFile >> number_of_velocities;

				// Go through the velocities
				for (int i = 0; i < number_of_velocities; i++)
				{
					// Create variables for the velocity
					double3 global_velocity = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> global_velocity.x;
					MyFile >> global_velocity.y;
					MyFile >> global_velocity.z;

					// Apply the global velocity on all nodes
					for (auto& node : nodes)
					{
						node.apply_fixed_velocity = 1;
						node.fixed_velocity = global_velocity;
						node.velocity = global_velocity;
					}
				}
			}

			// Check if the item is the COMPONENT keyword
			if (item == "COMPONENT")
			{
				// Read in the number of velocities we have to read
				MyFile >> number_of_velocities;

				// Go through the velocities
				for (int i = 0; i < number_of_velocities; i++)
				{
					// Create variables for the velocity
					int component = -1;
					double3 velocity = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> component;
					MyFile >> velocity.x;
					MyFile >> velocity.y;
					MyFile >> velocity.z;

					// Apply the velocity on the nodes that belong to the component
					for (auto& node : nodes)
					{
						if (node.component == component)
						{
							node.apply_fixed_velocity = 1;
							node.fixed_velocity = velocity;
							node.velocity = velocity;
						}
					}
				}
			}

			// Check if the item is the PLANE keyword
			if (item == "PLANE")
			{
				// Read in the number of velocities we have to read
				MyFile >> number_of_velocities;

				// Read the initial velocities
				for (int i = 0; i < number_of_velocities; i++)
				{
					// Create variable for the location
					double3 location = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> location.x;
					MyFile >> location.y;
					MyFile >> location.z;

					// Create variable for the plane normal
					double3 normal = make_double3(0.0, 0.0, 0.0);

					// Read the normal
					MyFile >> normal.x;
					MyFile >> normal.y;
					MyFile >> normal.z;

					// Create varible for the velocity
					double3 velocity = make_double3(0.0, 0.0, 0.0);

					// Read the velocity
					MyFile >> velocity.x;
					MyFile >> velocity.y;
					MyFile >> velocity.z;

					// Go through the nodes and find the ones that lay on the plane
					for (int i = 0; i < nodes.size(); i++)
					{
						// Check if the node is on the plane
						int node_on_plane = isNodeOnPlane(&nodes[0],
							i,
							location,
							normal);

						if (node_on_plane == 1)
						{
							// Apply the velocity onto the node
							nodes[i].apply_fixed_velocity = 1;
							nodes[i].fixed_velocity = velocity;
							nodes[i].velocity = velocity;
						}
					}
				}
			}

			// Check if the item is the LOCAL keyword
			if (item == "LOCAL")
			{
				// Read in the number of velocities we have to read
				MyFile >> number_of_velocities;

				// Go through the velocities
				for (int i = 0; i < number_of_velocities; i++)
				{
					// Create variables for the velocities
					double3 location = make_double3(0.0, 0.0, 0.0);
					double3 local_velocity = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> location.x;
					MyFile >> location.y;
					MyFile >> location.z;
					MyFile >> local_velocity.x;
					MyFile >> local_velocity.y;
					MyFile >> local_velocity.z;

					// Find the node closest to the location
					int node_index = findClosestNodeToLocation(&nodes[0],
															   nodes.size(),
															   location);

					// Apply the local initial condition on the node
					nodes[node_index].apply_fixed_velocity = 1;
					nodes[node_index].fixed_velocity = local_velocity; 
					nodes[node_index].velocity = local_velocity;
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}
