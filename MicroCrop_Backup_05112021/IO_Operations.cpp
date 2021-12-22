#include "Simulation.h"


// Load settings from TXT file
__host__ void loadSettings(Settings& settings)
{
	// Open the material peropeties file
	std::ifstream MyFile("0_INPUT/Settings.txt");

	// Create a string for the items
	std::string item;

	// Reads the file until it ends
	while (MyFile >> item)
	{
		// Check if the item is the MATERIALS keyword
		if (item == "MATERIALS")
		{
			// Variable for the number of material files
			int number_of_material_files = 0;

			// Read in the number of materials we have to read
			MyFile >> number_of_material_files;

			// Read the material peroperties
			for (int i = 0; i < number_of_material_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the file name
				MyFile >> file_name;

				// Add it to the settings
				settings.MaterialFiles.push_back(file_name);
			}
		}

		// Check if the item is the PARTICLES keyword
		if (item == "PARTICLES")
		{
			// Variable for the number of particle files
			int number_of_particle_files = 0;

			// Read in the number of materials we have to read
			MyFile >> number_of_particle_files;

			// Read the material peroperties
			for (int i = 0; i < number_of_particle_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the file name
				MyFile >> file_name;

				// Add it to the settings
				settings.ParticleFiles.push_back(file_name);
			}
		}

		// Check if the item is the INITIAL_CONDITION keyword
		if (item == "INITIAL_CONDITION")
		{
			// Variable for the number of initial condition files
			int number_of_initial_condition_files = 0;

			// Read in the number of initial condition files we have to read
			MyFile >> number_of_initial_condition_files;

			// Read the file names
			for (int i = 0; i < number_of_initial_condition_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the next file name
				MyFile >> file_name;

				// Add it to the settings
				settings.InitialConditionFiles.push_back(file_name);
			}
		}

		// Check if the item is the BOUNDARY_CONDITION keyword
		if (item == "BOUNDARY_CONDITION")
		{
			// Variable for the number of boundary condition files
			int number_of_boundary_condition_files = 0;

			// Read in the number of boundary condition files we have to read
			MyFile >> number_of_boundary_condition_files;

			// Read the file names
			for (int i = 0; i < number_of_boundary_condition_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the next file name
				MyFile >> file_name;

				// Add it to the settings
				settings.BoundaryConditionFiles.push_back(file_name);
			}
		}

		// Check if the item is the EXTERNAL_FORCE keyword
		if (item == "EXTERNAL_FORCE")
		{
			// Variable for the number of external force files
			int number_of_external_force_files = 0;

			// Read in the number of external force files we have to read
			MyFile >> number_of_external_force_files;

			// Read the file names
			for (int i = 0; i < number_of_external_force_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the next file name
				MyFile >> file_name;

				// Add it to the settings
				settings.ExternalForceFiles.push_back(file_name);
			}
		}

		// Check if the item is the OUTPUTFOLDER keyword
		if (item == "OUTPUTFOLDER")
		{
			// Read the output folder into the settings
			MyFile >> settings.OutputFolder;
		}

		// Check if the item is the NUMBER_OF_CPU_THREADS keyword
		if (item == "NUMBER_OF_CPU_THREADS")
		{
			// Read the number of CPU threads into the settings
			MyFile >> settings.number_of_CPU_threads;
		}

		// Check if the item is the START_TIME keyword
		if (item == "START_TIME")
		{
			// Read the start time for the simulation
			MyFile >> settings.start_time;
		}

		// Check if the item is the END_TIME keyword
		if (item == "END_TIME")
		{
			// Read the end time for the simulation
			MyFile >> settings.end_time;
		}

		// Check if the item is the TIMESTEP keyword
		if (item == "TIMESTEP")
		{
			// Read the timestep for the time integration
			MyFile >> settings.timestep;
		}

		// Check if the item is the SAVE_INTERVAL keyword
		if (item == "SAVE_INTERVAL")
		{
			// Read the save interval for the simulation
			MyFile >> settings.save_interval;
		}
	}

	// Close the file
	MyFile.close();
}


// Load crop material properties from TXT file
__host__ void loadMaterialProperties(Settings&			settings, 
									 MaterialContainer& materials)
{
	// Go through the material file names
	for (auto& file_name : settings.MaterialFiles)
	{
		// Open the material peropeties file
		std::ifstream MyFile(file_name);

		// Create a string for the items
		std::string item;

		// Variable for the number of materials
		int number_of_materials = 0;

		// Reads the file until it ends
		while (MyFile >> item)
		{
			// Check if the item is the NUMBER keyword
			if (item == "NUMBER")
			{
				// Read in the number of materials we have to process
				MyFile >> number_of_materials;

				// Create the material properties
				for (int i = 0; i < number_of_materials; i++)
				{
					// Create a new material instance with the given properties
					MaterialProperty new_material;

					// Create ID for the new material
					new_material.id = _material_id++;

					// Add the material to the material container
					materials.push_back(new_material);
				}
			}

			// Check if the item is the TYPE keyword
			if (item == "TYPE")
			{
				// Read the material types
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].type;
				}
			}

			// Check if the item is the LOCATION keyword
			if (item == "LOCATION")
			{
				// Read the locations
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].location.x;
					MyFile >> materials[i].location.y;
					MyFile >> materials[i].location.z;
				}
			}

			// Check if the item is the DENSITY keyword
			if (item == "DENSITY")
			{
				// Read the densities
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].density;
				}
			}

			// Check if the item is the ANISOTROPY_AXIS keyword
			if (item == "ANISOTROPY_AXIS")
			{
				// Read the anisotropy axes
				for (int i = 0; i < number_of_materials; i++)
				{
					// Go through the axes
					for (int j = 0; j < 3; j++)
					{
						MyFile >> materials[i].axes[j].x;
						MyFile >> materials[i].axes[j].y;
						MyFile >> materials[i].axes[j].z;
					}	
				}
			}

			// Check if the item is the ANISOTROPY_STIFFNESS keyword
			if (item == "ANISOTROPY_STIFFNESS")
			{
				// Read the anisotropy stiffnesses
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].axial_stiffnesses[0];
					MyFile >> materials[i].axial_stiffnesses[1];
					MyFile >> materials[i].axial_stiffnesses[2];
				}
			}

			// Check if the item is the ANISOTROPY_DAMPING keyword
			if (item == "ANISOTROPY_DAMPING")
			{
				// Read the anisotropy dampings
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].axial_dampings[0];
					MyFile >> materials[i].axial_dampings[1];
					MyFile >> materials[i].axial_dampings[2];
				}
			}

			// Check if the item is the ANISOTROPY_ROT_STIFFNESS keyword
			if (item == "ANISOTROPY_ROT_STIFFNESS")
			{
				// Read the anisotropy rotational stiffnesses
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].rotational_stiffnesses[0];
					MyFile >> materials[i].rotational_stiffnesses[1];
					MyFile >> materials[i].rotational_stiffnesses[2];
				}
			}

			// Check if the item is the ANISOTROPY_SPRING_STRENGTH keyword
			if (item == "ANISOTROPY_SPRING_STRENGTH")
			{
				// Read the anisotropy spring strengths
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> materials[i].strength[0];
					MyFile >> materials[i].strength[1];
					MyFile >> materials[i].strength[2];
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}

// Load particles
__host__ void loadNodesParticles(Settings&				settings, 
								 ParticleNodeContainer& nodes, 
								 ParticleContainer&		particles)
{
	// Go through the particle file names
	for (auto& file_name : settings.ParticleFiles)
	{
		// Open the VTK file
		std::ifstream VTKfile(file_name);

		// Create a temporary variable
		std::string item;

		// Create a vector for the points
		std::vector<double3> points;

		// Read the VTK file till its end
		while (VTKfile >> item)
		{
			// Check if the item is the POINTS keyword
			if (item == "POINTS")
			{
				// Read the number of points
				int number_of_points;
				VTKfile >> number_of_points;

				// Read the datatype into the temporary variable (we do not need this)
				VTKfile >> item;

				// Read the points one by one
				for (int i = 0; i < number_of_points; i++)
				{
					// Read the X-Y-Z coordinates of the points
					double x = 0.0;
					double y = 0.0;
					double z = 0.0;
					VTKfile >> x;
					VTKfile >> y;
					VTKfile >> z;

					// Create a new point and put it into the points container
					double3 new_point = make_double3(x, y, z);
					points.push_back(new_point);
				}
			}

			// Check if the item is the CELLS keyword
			if (item == "CELLS")
			{
				// Read the number of particles
				int number_of_particles;
				VTKfile >> number_of_particles;

				// Read the total number of data elements (we do not need this)
				VTKfile >> item;

				// Read the particles one by one
				for (int i = 0; i < number_of_particles; i++)
				{
					// Read the number of nodes
					int number_of_nodes = 0;
					VTKfile >> number_of_nodes;

					// Check if the cell is a tetrahedron
					if (number_of_nodes == 4)
					{
						// Create a new particle
						Particle new_particle;

						// Assign an ID to the particle
						new_particle.id = _particle_id++;

						// Set the status to be active
						new_particle.status = 1;

						// Read the point indices one by one
						for (int j = 0; j < number_of_nodes; j++)
						{
							// Read the point index
							int point_index = 0;
							VTKfile >> point_index;

							// Get the point location
							double3 location = points[point_index];

							// Get the node index that is in the same location
							int node_index = -1;
							if (nodes.size() > 0)
							{
								node_index = findParticleNodeInLocation(&nodes[0],
																		nodes.size(),
																		location);
							}

							// Check if the node exists
							if (node_index != -1)
							{
								// Link the new node to the particle
								new_particle.nodes[j] = node_index;
							}
							else
							{
								// Create a new particle node
								ParticleNode new_node;

								// Assign an ID to the new node
								new_node.id = _node_id++;

								// Assign position to the new node
								new_node.position = location;

								// Add the node to the node container
								nodes.push_back(new_node);

								// Link the new node to the particle
								new_particle.nodes[j] = nodes.size() - 1;
							}
						}

						// Add the new particle to the particle container
						particles.push_back(new_particle);
					}
					else
					{
						// The cell is not a tetrahedron, we simply read and skip the indices
						// Read the point indices one by one
						for (int j = 0; j < number_of_nodes; j++)
						{
							VTKfile >> item;
						}
					}
				}
			}
		}

		// Close the VTK file
		VTKfile.close();
	}
}







__host__ void writeParticleNodes(ParticleNodeContainer	nodes,
								 Settings				settings,
								 unsigned int			step)
{
	// Opening a VTK file
	std::string file_name = settings.OutputFolder + "ParticleNodes_Output_" + std::to_string(step) + ".vtk";
	std::ofstream MyFile(file_name);

	// Initial calculations
	int total_number_of_nodes = nodes.size();

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_nodes << " double\n";

	for (auto& node : nodes)
	{
		// Get the node position
		MyFile << node.position.x << " " << node.position.y << " " << node.position.z << "\n";
	}
	MyFile << "\n";

	// Writing the cells (particles)
	MyFile << "CELLS " << total_number_of_nodes << " " << total_number_of_nodes * 2 << "\n";

	for (int i = 0; i < total_number_of_nodes; i++)
	{
		MyFile << 1 << " " << i << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_nodes << "\n";

	for (int i = 0; i < total_number_of_nodes; i++)
	{
		MyFile << "1\n";
	}

	MyFile << "\n";

	// Writing point data
	MyFile << "POINT_DATA " << total_number_of_nodes << "\n";
	MyFile << "FIELD FieldData " << 5 << "\n";
	MyFile << "\n";

	MyFile << "NodeID_(#) " << 1 << " " << total_number_of_nodes << " double\n";
	for (auto& node : nodes)
	{
		MyFile << node.id << "\n";
	}
	MyFile << "\n";

	MyFile << "NodalMass_(kg) " << 1 << " " << total_number_of_nodes << " double\n";
	for (auto& node : nodes)
	{
		MyFile << node.mass << "\n";
	}
	MyFile << "\n";

	MyFile << "NodalForce_(N) " << 3 << " " << total_number_of_nodes << " double\n";
	for (auto& node : nodes)
	{
		MyFile << node.force.x << " " << node.force.y << " " << node.force.z << "\n";
	}
	MyFile << "\n";

	MyFile << "NodalVelocity_(m/s) " << 3 << " " << total_number_of_nodes << " double\n";
	for (auto& node : nodes)
	{
		MyFile << node.velocity.x << " " << node.velocity.y << " " << node.velocity.z << "\n";
	}
	MyFile << "\n";

	MyFile << "NodalAcceleration_(m/s2) " << 3 << " " << total_number_of_nodes << " double\n";
	for (auto& node : nodes)
	{
		MyFile << node.acceleration.x << " " << node.acceleration.y << " " << node.acceleration.z << "\n";
	}
	MyFile << "\n";

	// Closing the VTK file
	MyFile.close();
}




__host__ void writeParticleFaces(ParticleFaceContainer	faces, 
								 ParticleNodeContainer	nodes, 
								 Settings				settings, 
								 unsigned int			step)
{
	// Opening a VTK file
	std::string file_name = settings.OutputFolder + "ParticleFaces_Output_" + std::to_string(step) + ".vtk";
	std::ofstream MyFile(file_name);

	// Initial calculations
	int total_number_of_faces = faces.size();
	int total_number_of_face_points = total_number_of_faces * 3;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_face_points << " double\n";

	for (auto& face : faces)
	{
		for (int i = 0; i < 3; i++)
		{
			// Get the node position
			double3 position = nodes[face.nodes[i]].position;
			MyFile << position.x << " " << position.y << " " << position.z << "\n";
		}
	}
	MyFile << "\n";

	// Writing the cells (particles)
	MyFile << "CELLS " << total_number_of_faces << " " << total_number_of_faces * 4 << "\n";

	for (int i = 0; i < total_number_of_faces; i++)
	{
		int offset = i * 3;
		MyFile << 3 << " " << 0 + offset << " " << 1 + offset << " " << 2 + offset << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_faces << "\n";

	for (int i = 0; i < total_number_of_faces; i++)
	{
		MyFile << "5\n";
	}

	MyFile << "\n";

	// Writing point data
	MyFile << "POINT_DATA " << total_number_of_face_points << "\n";
	MyFile << "FIELD FieldData " << 4 << "\n";
	MyFile << "\n";

	MyFile << "FaceID_(#) " << 1 << " " << total_number_of_face_points << " double\n";
	for (auto& face : faces)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << face.id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "NodeID_(#) " << 1 << " " << total_number_of_face_points << " double\n";
	for (auto& face : faces)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << nodes[face.nodes[i]].id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "FaceArea_(m2) " << 1 << " " << total_number_of_face_points << " double\n";
	for (auto& face : faces)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << face.area << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "FaceNormal_(-) " << 3 << " " << total_number_of_face_points << " double\n";
	for (auto& face : faces)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << face.normal.x << " " << face.normal.y << " " << face.normal.z << "\n";
		}
	}
	MyFile << "\n";

	// Closing the VTK file
	MyFile.close();
}




__host__ void writeParticles(ParticleContainer		particles,
							 ParticleNodeContainer	nodes,
							 Settings				settings,
							 unsigned int			step)
{
	// Opening a VTK file
	std::string file_name = settings.OutputFolder + "Particles_Output_" + std::to_string(step) + ".vtk";
	std::ofstream MyFile(file_name);

	// Initial calculations
	int total_number_of_particles = particles.size();
	int total_number_of_points = total_number_of_particles * 4;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			// Get the node position
			double3 position = nodes[particle.nodes[i]].position;
			MyFile << position.x << " " << position.y << " " << position.z << "\n";
		}
	}
	MyFile << "\n";

	// Writing the cells (particles)
	MyFile << "CELLS " << total_number_of_particles << " " << total_number_of_particles * 5 << "\n";

	for (int i = 0; i < total_number_of_particles; i++)
	{
		int offset = i * 4;
		MyFile << 4 << " " << 0 + offset << " " << 1 + offset << " " << 2 + offset << " " << 3 + offset << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_particles << "\n";

	for (int i = 0; i < total_number_of_particles; i++)
	{
		MyFile << "10\n";
	}

	MyFile << "\n";

	// Writing point data
	MyFile << "POINT_DATA " << total_number_of_points << "\n";
	MyFile << "FIELD FieldData " << 7 << "\n";
	MyFile << "\n";

	MyFile << "ParticleID_(#) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			MyFile << particle.id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "ParticleStatus_(-) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			MyFile << particle.status << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "NodeID_(#) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			MyFile << nodes[particle.nodes[i]].id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "NodalMass_(kg) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			MyFile << nodes[particle.nodes[i]].mass << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "NodalForce_(N) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			double3 force = nodes[particle.nodes[i]].force;
			MyFile << force.x << " " << force.y << " " << force.z << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "NodalVelocity_(m/s) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			double3 velocity = nodes[particle.nodes[i]].velocity;
			MyFile << velocity.x << " " << velocity.y << " " << velocity.z << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "NodalAcceleration_(m/s2) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 4; i++)
		{
			double3 acceleration = nodes[particle.nodes[i]].acceleration;
			MyFile << acceleration.x << " " << acceleration.y << " " << acceleration.z << "\n";
		}
	}
	MyFile << "\n";

	// Closing the VTK file
	MyFile.close();
}

__host__ void wrtieParticleAxialSprings(AxialSpringContainer		axial_springs, 
										ParticleContainer			particles, 
										IntersectionNodeContainer	intersection_nodes, 
										Settings					settings, 
										unsigned int				step)
{
	// Opening a VTK file
	std::string file_name = settings.OutputFolder + "ParticleAxialSprings_Output_" + std::to_string(step) + ".vtk";
	std::ofstream MyFile(file_name);

	// Initial calculations
	int total_number_of_springs = particles.size() * 3;
	int total_number_of_points = total_number_of_springs * 2;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& particle : particles)
	{
		for (int i = 0; i < 6; i++)
		{
			double3 position = intersection_nodes[particle.intersections[i]].position;
			MyFile << position.x << " " << position.y << " " << position.z << "\n";
		}
	}
	MyFile << "\n";

	// Writing the cells (particles)
	MyFile << "CELLS " << total_number_of_springs << " " << total_number_of_springs * 3 << "\n";

	for (int i = 0; i < total_number_of_springs; i++)
	{
		int offset = i * 2;
		MyFile << 2 << " " << 0 + offset << " " << 1 + offset << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_springs << "\n";

	for (int i = 0; i < total_number_of_springs; i++)
	{
		MyFile << "3\n";
	}

	MyFile << "\n";

	//Writing point data
	MyFile << "POINT_DATA " << total_number_of_points << "\n";
	MyFile << "FIELD FieldData " << 8 << "\n";
	MyFile << "\n";

	MyFile << "ParticleID_(#) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 6; i++)
		{
			MyFile << particle.id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringID_(#) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << axial_springs[particle.axial_springs[i]].id << "\n";
			MyFile << axial_springs[particle.axial_springs[i]].id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringStiffness_(N/m) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << axial_springs[particle.axial_springs[i]].stiffness << "\n";
			MyFile << axial_springs[particle.axial_springs[i]].stiffness << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringDamping_(Ns/m) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << axial_springs[particle.axial_springs[i]].damping << "\n";
			MyFile << axial_springs[particle.axial_springs[i]].damping << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringLength_(m) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << axial_springs[particle.axial_springs[i]].current_length << "\n";
			MyFile << axial_springs[particle.axial_springs[i]].current_length << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringForce_(N) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			double3 force_a = axial_springs[particle.axial_springs[i]].spring_force_node_a;
			double3 force_b = axial_springs[particle.axial_springs[i]].spring_force_node_b;
			MyFile << force_a.x << " " << force_a.y << " " << force_a.z << "\n";
			MyFile << force_b.x << " " << force_b.y << " " << force_b.z << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "DampingForce_(N) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			double3 force_a = axial_springs[particle.axial_springs[i]].damping_force_node_a;
			double3 force_b = axial_springs[particle.axial_springs[i]].damping_force_node_b;
			MyFile << force_a.x << " " << force_a.y << " " << force_a.z << "\n";
			MyFile << force_b.x << " " << force_b.y << " " << force_b.z << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "TotalForce_(N) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			double3 force_a = axial_springs[particle.axial_springs[i]].total_force_node_a;
			double3 force_b = axial_springs[particle.axial_springs[i]].total_force_node_b;
			MyFile << force_a.x << " " << force_a.y << " " << force_a.z << "\n";
			MyFile << force_b.x << " " << force_b.y << " " << force_b.z << "\n";
		}
	}
	MyFile << "\n";

	// Closing the VTK file
	MyFile.close();
}



__host__ void writeParticleRotationalSprings(ParticleContainer			particles,
	RotationalSpringContainer	rotational_springs,
	AxialSpringContainer		axial_springs,
	IntersectionNodeContainer	intersection_nodes,
	Settings					settings,
	unsigned int				step)
{
	// Opening a VTK file
	std::string file_name = settings.OutputFolder + "ParticleRotationalSprings_Output_" + std::to_string(step) + ".vtk";
	std::ofstream MyFile(file_name);

	// Initial calculations
	int total_number_of_springs = particles.size() * 3;
	int total_number_of_points = total_number_of_springs * 4;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			int rspring_index = particle.rotational_springs[i];

			int aspring_a_index = rotational_springs[rspring_index].axial_springs[0];
			int aspring_b_index = rotational_springs[rspring_index].axial_springs[1];

			int aspring_a_node_a_index = axial_springs[aspring_a_index].nodes[0];
			int aspring_a_node_b_index = axial_springs[aspring_a_index].nodes[1];
			int aspring_b_node_a_index = axial_springs[aspring_b_index].nodes[0];
			int aspring_b_node_b_index = axial_springs[aspring_b_index].nodes[1];

			double3 aspring_a_node_a_position = intersection_nodes[aspring_a_node_a_index].position;
			double3 aspring_a_node_b_position = intersection_nodes[aspring_a_node_b_index].position;
			double3 aspring_b_node_a_position = intersection_nodes[aspring_b_node_a_index].position;
			double3 aspring_b_node_b_position = intersection_nodes[aspring_b_node_b_index].position;

			MyFile << aspring_a_node_a_position.x << " " << aspring_a_node_a_position.y << " " << aspring_a_node_a_position.z << "\n";
			MyFile << aspring_b_node_a_position.x << " " << aspring_b_node_a_position.y << " " << aspring_b_node_a_position.z << "\n";
			MyFile << aspring_a_node_b_position.x << " " << aspring_a_node_b_position.y << " " << aspring_a_node_b_position.z << "\n";
			MyFile << aspring_b_node_b_position.x << " " << aspring_b_node_b_position.y << " " << aspring_b_node_b_position.z << "\n";
		}
	}
	MyFile << "\n";

	// Writing the cells (particles)
	MyFile << "CELLS " << total_number_of_springs << " " << total_number_of_springs * 5 << "\n";

	for (int i = 0; i < total_number_of_springs; i++)
	{
		int offset = i * 4;
		MyFile << 4 << " " << 0 + offset << " " << 1 + offset << " " << 2 + offset << " " << 3 + offset << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_springs << "\n";

	for (int i = 0; i < total_number_of_springs; i++)
	{
		MyFile << "9\n";
	}

	MyFile << "\n";

	//Writing point data
	MyFile << "POINT_DATA " << total_number_of_points << "\n";
	MyFile << "FIELD FieldData " << 5 << "\n";
	MyFile << "\n";

	MyFile << "ParticleID_(#) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 12; i++)
		{
			MyFile << particle.id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringID_(#) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << rotational_springs[particle.rotational_springs[i]].id << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].id << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].id << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].id << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringStiffness_(N/rad) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << rotational_springs[particle.rotational_springs[i]].stiffness << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].stiffness << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].stiffness << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].stiffness << "\n";
		}
	}
	MyFile << "\n";


	MyFile << "SpringAngle_(rad) " << 1 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			MyFile << rotational_springs[particle.rotational_springs[i]].current_angle << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].current_angle << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].current_angle << "\n";
			MyFile << rotational_springs[particle.rotational_springs[i]].current_angle << "\n";
		}
	}
	MyFile << "\n";

	MyFile << "SpringForce_(N) " << 3 << " " << total_number_of_points << " double\n";
	for (auto& particle : particles)
	{
		for (int i = 0; i < 3; i++)
		{
			double3 force_a = rotational_springs[particle.rotational_springs[i]].spring_a_node_a_force;
			double3 force_b = rotational_springs[particle.rotational_springs[i]].spring_a_node_b_force;
			double3 force_c = rotational_springs[particle.rotational_springs[i]].spring_b_node_a_force;
			double3 force_d = rotational_springs[particle.rotational_springs[i]].spring_b_node_b_force;

			MyFile << force_a.x << " " << force_a.y << " " << force_a.z << "\n";
			MyFile << force_c.x << " " << force_c.y << " " << force_c.z << "\n";
			MyFile << force_b.x << " " << force_b.y << " " << force_b.z << "\n";
			MyFile << force_d.x << " " << force_d.y << " " << force_d.z << "\n";
		}
	}
	MyFile << "\n";

	// Closing the VTK file
	MyFile.close();
}