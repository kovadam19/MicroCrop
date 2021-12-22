#include "Simulation.h"

__host__ void applyInitialConditions(ParticleNodeContainer& nodes,
									 Settings& settings)
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

				// Read the material peroperties
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

			// Check if the item is the LOCAL keyword
			if (item == "LOCAL")
			{
				// Read in the number of conditions we have to read
				MyFile >> number_of_conditions;

				// Read the material peroperties
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
					int node_index = findClosestParticleNodeToLocation(&nodes[0],
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