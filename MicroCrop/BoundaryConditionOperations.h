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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _                      _             ____                                _                             ____                       _   _   _     _                       
//      / \     _ __    _ __   | |  _   _    | __ )    ___    _   _   _ __     __| |   __ _   _ __   _   _     / ___|   ___    _ __     __| | (_) | |_  (_)   ___    _ __    ___ 
//     / _ \   | '_ \  | '_ \  | | | | | |   |  _ \   / _ \  | | | | | '_ \   / _` |  / _` | | '__| | | | |   | |      / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \  / __|
//    / ___ \  | |_) | | |_) | | | | |_| |   | |_) | | (_) | | |_| | | | | | | (_| | | (_| | | |    | |_| |   | |___  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | | \__ \
//   /_/   \_\ | .__/  | .__/  |_|  \__, |   |____/   \___/   \__,_| |_| |_|  \__,_|  \__,_| |_|     \__, |    \____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_| |___/
//             |_|     |_|          |___/                                                            |___/                                                                       
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Parses the boundary condition input file and applies the given boundary conditions
inline __host__ void applyBoundaryConditions(NodeContainer&		nodes,
											 Settings&			settings)
{
	// Go through the boundary condition file names
	for (auto& file_name : settings.BoundaryConditionFiles)
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

				// Read the boundary conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create variables for the boundary conditions
					int3 boundary = make_int3(0, 0, 0);

					// Read boundary conditions
					MyFile >> boundary.x;
					MyFile >> boundary.y;
					MyFile >> boundary.z;

					// Apply the boundary condition on all nodes
					for (auto& node : nodes)
					{
						node.boundaries = boundary;
					}
				}
			}

			// Check if the item is the COMPONENT keyword
			if (item == "COMPONENT")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Read the boundary conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create a variable for the component
					int component = -1;

					// Read the component
					MyFile >> component;

					// Create variables for the boundary conditions
					int3 boundary = make_int3(0, 0, 0);

					// Read boundary conditions
					MyFile >> boundary.x;
					MyFile >> boundary.y;
					MyFile >> boundary.z;

					// Apply the boundary condition on all nodes that belong to the component
					for (auto& node : nodes)
					{
						if (node.component == component)
						{
							nodes[i].boundaries = boundary;
						}
					}
				}
			}

			// Check if the item is the PLANE keyword
			if (item == "PLANE")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Read the boundary conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create a variable for the location
					double3 location = make_double3(0.0, 0.0, 0.0);

					// Read the location
					MyFile >> location.x;
					MyFile >> location.y;
					MyFile >> location.z;

					// Create a variable for the plane normal
					double3 normal = make_double3(0.0, 0.0, 0.0);

					// Read the plane normal
					MyFile >> normal.x;
					MyFile >> normal.y;
					MyFile >> normal.z;

					// Create variables for the boundary conditions
					int3 boundary = make_int3(0, 0, 0);

					// Read boundary conditions
					MyFile >> boundary.x;
					MyFile >> boundary.y;
					MyFile >> boundary.z;

					// Apply the boundary condition on all nodes that are on the plane
					for (int j = 0; j < nodes.size(); j++)
					{
						// Check if the node is on the plane
						int point_on_plane = isNodeOnPlane(&nodes[0],
														   j,
														   location,
														   normal);

						if (point_on_plane == 1)
						{
							nodes[j].boundaries = boundary;
						}
					}
				}
			}

			// Check if the item is the LOCAL keyword
			if (item == "LOCAL")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Read the boundary conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create variables for the boundary condition
					double3 location = make_double3(0.0, 0.0, 0.0);
					int3 boundaries = make_int3(0, 0, 0);

					// Read the values from the file
					MyFile >> location.x;
					MyFile >> location.y;
					MyFile >> location.z;
					MyFile >> boundaries.x;
					MyFile >> boundaries.y;
					MyFile >> boundaries.z;

					// Find the node closest to the location
					int node_index = findClosestNodeToLocation(&nodes[0],
															   nodes.size(),
															   location);

					// Apply the local boundary condition on the node
					if (node_index != -1)
					{
						nodes[node_index].boundaries = boundaries;
					}
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}