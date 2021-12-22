#include "Simulation.h"

__host__ void applyBoundaryConditions(NodeContainer&	nodes,
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
			// Check if the item is the POINT keyword
			if (item == "POINT")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Read the boundary conditions
				for (int i = 0; i < number_of_conditions; i++)
				{
					// Create variables for the initial condition
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

					// Apply the local initial condition on the node
					if (node_index != -1)
					{
						nodes[node_index].boundaries = boundaries;
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
					for (int i = 0; i < nodes.size(); i++)
					{
						// Check if the node is on the plane
						int point_on_plane = isNodeOnPlane(&nodes[0],
																	i,
																	location,
																	normal);

						if (point_on_plane == 1)
						{
							nodes[i].boundaries = boundary;
						}
					}
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}