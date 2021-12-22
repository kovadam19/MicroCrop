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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _                      _             ___           _   _     _           _      ____                       _   _   _     _                 
//      / \     _ __    _ __   | |  _   _    |_ _|  _ __   (_) | |_  (_)   __ _  | |    / ___|   ___    _ __     __| | (_) | |_  (_)   ___    _ __  
//     / _ \   | '_ \  | '_ \  | | | | | |    | |  | '_ \  | | | __| | |  / _` | | |   | |      / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \ 
//    / ___ \  | |_) | | |_) | | | | |_| |    | |  | | | | | | | |_  | | | (_| | | |   | |___  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | |
//   /_/   \_\ | .__/  | .__/  |_|  \__, |   |___| |_| |_| |_|  \__| |_|  \__,_| |_|    \____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_|
//             |_|     |_|          |___/                                                                                                           
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Parses the input files and applies the initial conditions onto the nodes
inline __host__ void applyInitialConditions(NodeContainer&	nodes,
											Settings&		settings)
{
	// Go through the initial condition file names
	for (auto& file_name : settings.InitialConditionFiles)
	{
		// Open the current file
		std::ifstream MyFile(file_name);

		// Create a string for the items
		std::string item;

		// Variable for the number of conditions
		int number_of_conditions = 0;

		// Reads the file until it ends
		while (MyFile >> item)
		{
			// Check if the item is the GLOBAL keyword
			if (item == "GLOBAL")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Go through the conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create variables for the initial condition
					double3 global_velocity = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> global_velocity.x;
					MyFile >> global_velocity.y;
					MyFile >> global_velocity.z;

					// Apply the global initial condition on all nodes
					for (auto& node : nodes)
					{
						node.velocity = global_velocity;
					}
				}
			}

			// Check if the item is the COMPONENT keyword
			if (item == "COMPONENT")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Go through the conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create variables for the initial condition
					int component = -1;
					double3 velocity = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> component;
					MyFile >> velocity.x;
					MyFile >> velocity.y;
					MyFile >> velocity.z;

					// Apply the global initial condition on the nodes that belong to the component
					for (auto& node : nodes)
					{
						if (node.component == component)
						{
							node.velocity = velocity;
						}	
					}
				}
			}

			// Check if the item is the PLANE keyword
			if (item == "PLANE")
			{
				// Read in the number of initial conditions we have to read
				MyFile >> number_of_conditions;

				// Read the initial conditions
				for (int i = 0; i < number_of_conditions; i++)
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
							// Apply the initial conidtion onto the node
							nodes[i].velocity = velocity;
						}
					}
				}
			}

			// Check if the item is the LOCAL keyword
			if (item == "LOCAL")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Go through the conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create variables for the initial condition
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
					nodes[node_index].velocity = local_velocity;
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}
