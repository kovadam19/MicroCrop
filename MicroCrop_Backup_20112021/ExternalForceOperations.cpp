#include "Simulation.h"

__host__ void createExternalForces(ExternalForceContainer&	external_forces,
								   NodeContainer&			nodes,
								   Settings&				settings)
{
	// Go through the boundary condition file names
	for (auto& file_name : settings.ExternalForceFiles)
	{
		// Open the current file
		std::ifstream MyFile(file_name);

		// Create a string for the items
		std::string item;

		// Variable for the number of external forces
		int number_of_external_forces = 0;

		// Reads the file until it ends
		while (MyFile >> item)
		{
			// Check if the item is the LOCAL keyword
			if (item == "LOCAL")
			{
				// Read in the number of external forces we have to read
				MyFile >> number_of_external_forces;

				// Read the extrnal forces
				for (int i = 0; i < number_of_external_forces; i++)
				{
					// Create variable for the location
					double3 location = make_double3(0.0, 0.0, 0.0);

					// Read the values from the file
					MyFile >> location.x;
					MyFile >> location.y;
					MyFile >> location.z;

					// Create variables for the timing
					double start_time = 0.0;
					double duration = 0.0;

					// Read the timing variables
					MyFile >> start_time;
					MyFile >> duration;

					// Create varible for the force
					double3 force = make_double3(0.0, 0.0, 0.0);

					// Read the force
					MyFile >> force.x;
					MyFile >> force.y;
					MyFile >> force.z;

					// Find the node closest to the location
					int node_index = findClosestNodeToLocation(&nodes[0],
																	   nodes.size(),
																	   location);

					// Check if we found a node
					if (node_index != -1)
					{
						// Create an external force on the node
						ExternalForce new_eforce;

						// Assign an ID to the force
						new_eforce.id = _external_force_id++;

						// Assign the node to the force
						new_eforce.node = node_index;

						// Assign the start time to the force
						new_eforce.start_time = start_time;

						// Assign the duration to the force
						new_eforce.duration = duration;

						// Set the force
						new_eforce.force = force;

						// Add the new force to the container
						external_forces.push_back(new_eforce);
					}
				}
			}

			// Check if the item is the PLANE keyword
			if (item == "PLANE")
			{
				// Read in the number of external forces we have to read
				MyFile >> number_of_external_forces;

				// Read the extrnal forces
				for (int i = 0; i < number_of_external_forces; i++)
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

					// Create variables for the timing
					double start_time = 0.0;
					double duration = 0.0;

					// Read the timing variables
					MyFile >> start_time;
					MyFile >> duration;

					// Create varible for the force
					double3 force = make_double3(0.0, 0.0, 0.0);

					// Read the force
					MyFile >> force.x;
					MyFile >> force.y;
					MyFile >> force.z;

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
							// Create an external force on the node
							ExternalForce new_eforce;

							// Assign an ID to the force
							new_eforce.id = _external_force_id++;

							// Assign the node to the force
							new_eforce.node = i;

							// Assign the start time to the force
							new_eforce.start_time = start_time;

							// Assign the duration to the force
							new_eforce.duration = duration;

							// Set the force
							new_eforce.force = force;

							// Add the new force to the container
							external_forces.push_back(new_eforce);
						}
					}
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}




__host__ __device__ void applyExternalForces(ExternalForce* external_forces,
											 int	number_of_forces,
											 Node*  nodes,
											 double			time)
{
	// Go through the external forces
	for (int i = 0; i < number_of_forces; i++)
	{
		// Get the start time and the duration
		double start_time = external_forces[i].start_time;
		double duration = external_forces[i].duration;

		// Check if the external force is active at the current time
		if (start_time <= time && time <= (start_time + duration))
		{
			// Get the node index
			int node_index = external_forces[i].node;

			// Apply the force onto the node
			nodes[node_index].force += external_forces[i].force;
		}
	}
}