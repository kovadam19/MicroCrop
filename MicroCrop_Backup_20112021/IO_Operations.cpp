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

		// Check if the item is the CELLS keyword
		if (item == "CELLS")
		{
			// Variable for the number of cell files
			int number_of_cell_files = 0;

			// Read in the number of materials we have to read
			MyFile >> number_of_cell_files;

			// Read the material peroperties
			for (int i = 0; i < number_of_cell_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the file name
				MyFile >> file_name;

				// Add it to the settings
				settings.CellFiles.push_back(file_name);
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

		// Check if the item is the SIMULATION_ON_GPU keyword
		if (item == "SIMULATION_ON_GPU")
		{
			// Read the setting
			MyFile >> settings.simulation_on_GPU;
		}

		// Check if the item is the GPU_DEVICE keyword
		if (item == "GPU_DEVICE")
		{
			// Read the setting
			MyFile >> settings.GPU_device;
		}

		// Check if the item is the GPU_THREADS_PER_BLOCK keyword
		if (item == "GPU_THREADS_PER_BLOCK")
		{
			// Read the setting
			MyFile >> settings.GPU_threads_per_block;
		}

		// Check if the item is the GPU_NUMBER_OF_BLOCKS keyword
		if (item == "GPU_NUMBER_OF_BLOCKS")
		{
			// Read the setting
			MyFile >> settings.GPU_number_of_blocks;
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

		// Check if the item is the ADJUST_ANGLE_X keyword
		if (item == "ADJUST_ANGLE_X")
		{
			// Read the adjust angle for the simulation
			MyFile >> settings.adjust_angle_x;
		}

		// Check if the item is the ADJUST_ANGLE_Y keyword
		if (item == "ADJUST_ANGLE_Y")
		{
			// Read the adjust angle for the simulation
			MyFile >> settings.adjust_angle_y;
		}

		// Check if the item is the ADJUST_ANGLE_Z keyword
		if (item == "ADJUST_ANGLE_Z")
		{
			// Read the adjust angle for the simulation
			MyFile >> settings.adjust_angle_z;
		}

		// Check if the item is the SAVE_CELLS keyword
		if (item == "SAVE_CELLS")
		{
			// Read setting
			MyFile >> settings.save_cells;
		}

		// Check if the item is the SAVE_CELL_ID keyword
		if (item == "SAVE_CELL_ID")
		{
			// Read setting
			MyFile >> settings.save_cell_id;
		}

		// Check if the item is the SAVE_CELL_STATUS keyword
		if (item == "SAVE_CELL_STATUS")
		{
			// Read setting
			MyFile >> settings.save_cell_status;
		}

		// Check if the item is the SAVE_CELL_MATERIAL_TYPE keyword
		if (item == "SAVE_CELL_MATERIAL_TYPE")
		{
			// Read setting
			MyFile >> settings.save_cell_material_type;
		}

		// Check if the item is the SAVE_CELL_MATERIAL_PROPERTY keyword
		if (item == "SAVE_CELL_MATERIAL_PROPERTY")
		{
			// Read setting
			MyFile >> settings.save_cell_material_property;
		}

		// Check if the item is the SAVE_CELL_VOLUME keyword
		if (item == "SAVE_CELL_VOLUME")
		{
			// Read setting
			MyFile >> settings.save_cell_volume;
		}

		// Check if the item is the SAVE_CELL_MASS keyword
		if (item == "SAVE_CELL_MASS")
		{
			// Read setting
			MyFile >> settings.save_cell_mass;
		}

		// Check if the item is the SAVE_CELL_NODE_ID keyword
		if (item == "SAVE_CELL_NODE_ID")
		{
			// Read setting
			MyFile >> settings.save_cell_node_id;
		}

		// Check if the item is the SAVE_CELL_NODE_MASS keyword
		if (item == "SAVE_CELL_NODE_MASS")
		{
			// Read setting
			MyFile >> settings.save_cell_node_mass;
		}

		// Check if the item is the SAVE_CELL_NODE_FORCE keyword
		if (item == "SAVE_CELL_NODE_FORCE")
		{
			// Read setting
			MyFile >> settings.save_cell_node_force;
		}

		// Check if the item is the SAVE_CELL_NODE_VELOCITY keyword
		if (item == "SAVE_CELL_NODE_VELOCITY")
		{
			// Read setting
			MyFile >> settings.save_cell_node_velocity;
		}

		// Check if the item is the SAVE_CELL_NODE_ACCELERATION keyword
		if (item == "SAVE_CELL_NODE_ACCELERATION")
		{
			// Read setting
			MyFile >> settings.save_cell_node_acceleration;
		}

		// Check if the item is the SAVE_FACES keyword
		if (item == "SAVE_FACES")
		{
			// Read setting
			MyFile >> settings.save_faces;
		}

		// Check if the item is the SAVE_FACE_ID keyword
		if (item == "SAVE_FACE_ID")
		{
			// Read setting
			MyFile >> settings.save_face_id;
		}

		// Check if the item is the SAVE_FACE_CELL_ID keyword
		if (item == "SAVE_FACE_CELL_ID")
		{
			// Read setting
			MyFile >> settings.save_face_cell_id;
		}

		// Check if the item is the SAVE_FACE_AREA keyword
		if (item == "SAVE_FACE_AREA")
		{
			// Read setting
			MyFile >> settings.save_face_area;
		}

		// Check if the item is the SAVE_FACE_NORMAL keyword
		if (item == "SAVE_FACE_NORMAL")
		{
			// Read setting
			MyFile >> settings.save_face_normal;
		}

		// Check if the item is the SAVE_NODES keyword
		if (item == "SAVE_NODES")
		{
			// Read setting
			MyFile >> settings.save_nodes;
		}

		// Check if the item is the SAVE_NODE_ID keyword
		if (item == "SAVE_NODE_ID")
		{
			// Read setting
			MyFile >> settings.save_node_id;
		}

		// Check if the item is the SAVE_NODE_MASS keyword
		if (item == "SAVE_NODE_MASS")
		{
			// Read setting
			MyFile >> settings.save_node_mass;
		}

		// Check if the item is the SAVE_NODE_FORCE keyword
		if (item == "SAVE_NODE_FORCE")
		{
			// Read setting
			MyFile >> settings.save_node_force;
		}

		// Check if the item is the SAVE_NODE_VELOCITY keyword
		if (item == "SAVE_NODE_VELOCITY")
		{
			// Read setting
			MyFile >> settings.save_node_velocity;
		}

		// Check if the item is the SAVE_NODE_ACCELERATION keyword
		if (item == "SAVE_NODE_ACCELERATION")
		{
			// Read setting
			MyFile >> settings.save_node_acceleration;
		}

		// Check if the item is the SAVE_AXIAL_SPRINGS keyword
		if (item == "SAVE_AXIAL_SPRINGS")
		{
			// Read setting
			MyFile >> settings.save_axial_springs;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_ID keyword
		if (item == "SAVE_AXIAL_SPRING_ID")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_id;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_CELL_ID keyword
		if (item == "SAVE_AXIAL_SPRING_CELL_ID")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_cell_id;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_STIFFNESS keyword
		if (item == "SAVE_AXIAL_SPRING_STIFFNESS")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_stiffness;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_DAMPING keyword
		if (item == "SAVE_AXIAL_SPRING_DAMPING")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_damping;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_LENGTH keyword
		if (item == "SAVE_AXIAL_SPRING_LENGTH")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_length;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_SPRING_FORCE keyword
		if (item == "SAVE_AXIAL_SPRING_SPRING_FORCE")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_spring_force;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_DAMPING_FORCE keyword
		if (item == "SAVE_AXIAL_SPRING_DAMPING_FORCE")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_damping_force;
		}

		// Check if the item is the SAVE_AXIAL_SPRING_TOTAL_FORCE keyword
		if (item == "SAVE_AXIAL_SPRING_TOTAL_FORCE")
		{
			// Read setting
			MyFile >> settings.save_axial_spring_total_force;
		}

		// Check if the item is the SAVE_ROTATIONAL_SPRINGS keyword
		if (item == "SAVE_ROTATIONAL_SPRINGS")
		{
			// Read setting
			MyFile >> settings.save_rotational_springs;
		}

		// Check if the item is the SAVE_ROTATIONAL_SPRING_ID keyword
		if (item == "SAVE_ROTATIONAL_SPRING_ID")
		{
			// Read setting
			MyFile >> settings.save_rotational_spring_id;
		}

		// Check if the item is the SAVE_ROTATIONAL_SPRING_CELL_ID keyword
		if (item == "SAVE_ROTATIONAL_SPRING_CELL_ID")
		{
			// Read setting
			MyFile >> settings.save_rotational_spring_cell_id;
		}

		// Check if the item is the SAVE_ROTATIONAL_SPRING_STIFFNESS keyword
		if (item == "SAVE_ROTATIONAL_SPRING_STIFFNESS")
		{
			// Read setting
			MyFile >> settings.save_rotational_spring_stiffness;
		}

		// Check if the item is the SAVE_ROTATIONAL_SPRING_ANGLE keyword
		if (item == "SAVE_ROTATIONAL_SPRING_ANGLE")
		{
			// Read setting
			MyFile >> settings.save_rotational_spring_angle;
		}

		// Check if the item is the SAVE_ROTATIONAL_SPRING_FORCE keyword
		if (item == "SAVE_ROTATIONAL_SPRING_FORCE")
		{
			// Read setting
			MyFile >> settings.save_rotational_spring_force;
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

// Load cells
__host__ void loadCells(Settings&		settings, 
						NodeContainer&	nodes, 
						CellContainer&	cells)
{
	// Go through the cell file names
	for (auto& file_name : settings.CellFiles)
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
				std::cout << "Reading point data..." << std::endl;

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
				std::cout << "Reading cell data..." << std::endl;

				// Create two vectors to keep track of which nodes are already created and what is their location among the nodes
				std::vector<bool> point_is_created(points.size(), false);
				std::vector<int> point_location(points.size(), -1);

				// Read the number of cells
				int number_of_cells;
				VTKfile >> number_of_cells;

				// Read the total number of data elements (we do not need this)
				VTKfile >> item;

				// Read the cells one by one
				for (int i = 0; i < number_of_cells; i++)
				{
					// Read the number of nodes
					int number_of_nodes = 0;
					VTKfile >> number_of_nodes;

					// Check if the cell is a tetrahedron
					if (number_of_nodes == 4)
					{
						// Create a new cell
						Cell new_cell;

						// Assign an ID to the cell
						new_cell.id = _cell_id++;

						// Set the status to be active
						new_cell.status = 1;

						// Read the point indices one by one
						for (int j = 0; j < number_of_nodes; j++)
						{
							// Read the point index
							int point_index = 0;
							VTKfile >> point_index;

							// Check if the node exists
							if (point_is_created[point_index])
							{
								// Link the new node to the cell
								new_cell.nodes[j] = point_location[point_index];
							}
							else
							{
								// Create a new cell node
								Node new_node;

								// Assign an ID to the new node
								new_node.id = _node_id++;

								// Assign position to the new node
								new_node.position = points[point_index];

								// Add the node to the node container
								nodes.push_back(new_node);

								// Set the point creation to be true
								point_is_created[point_index] = true;

								// Get the location of the point among the nodes
								point_location[point_index] = nodes.size() - 1;

								// Link the new node to the cell
								new_cell.nodes[j] = nodes.size() - 1;
							}
						}

						// Add the new cell to the cell container
						cells.push_back(new_cell);
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







__host__ void writeCellNodes(NodeContainer	nodes,
							 Settings		settings,
							 int			step,
							 std::string	file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_nodes = nodes.size();
	int number_of_field_data = 0;

	if (settings.save_node_id == 1) number_of_field_data++;
	if (settings.save_node_mass == 1) number_of_field_data++;
	if (settings.save_node_force == 1) number_of_field_data++;
	if (settings.save_node_velocity == 1) number_of_field_data++;
	if (settings.save_node_acceleration == 1) number_of_field_data++;

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

	// Writing the cells (cells)
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
	if (number_of_field_data > 0)
	{
		MyFile << "POINT_DATA " << total_number_of_nodes << "\n";
		MyFile << "FIELD FieldData " << number_of_field_data << "\n";
		MyFile << "\n";

		if (settings.save_node_id == 1)
		{
			MyFile << "NodeID_(#) " << 1 << " " << total_number_of_nodes << " int\n";
			for (auto& node : nodes)
			{
				MyFile << node.id << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_node_mass == 1)
		{
			MyFile << "NodalMass_(kg) " << 1 << " " << total_number_of_nodes << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.mass << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_node_force == 1)
		{
			MyFile << "NodalForce_(N) " << 3 << " " << total_number_of_nodes << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.force.x << " " << node.force.y << " " << node.force.z << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_node_velocity == 1)
		{
			MyFile << "NodalVelocity_(m/s) " << 3 << " " << total_number_of_nodes << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.velocity.x << " " << node.velocity.y << " " << node.velocity.z << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_node_acceleration == 1)
		{
			MyFile << "NodalAcceleration_(m/s2) " << 3 << " " << total_number_of_nodes << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.acceleration.x << " " << node.acceleration.y << " " << node.acceleration.z << "\n";
			}
			MyFile << "\n";
		}
	}

	// Closing the VTK file
	MyFile.close();
}


__host__ void writeCellFaces(FaceContainer	faces, 
							 NodeContainer	nodes, 
							 Settings		settings, 
							 int			step,
							 std::string	file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_faces = faces.size();
	int total_number_of_points = nodes.size();

	int number_of_cell_data = 0;

	if (settings.save_face_id) number_of_cell_data++;
	if (settings.save_face_cell_id) number_of_cell_data++;
	if (settings.save_face_area) number_of_cell_data++;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& node : nodes)
	{
		MyFile << node.position.x << " " << node.position.y << " " << node.position.z << "\n";
	}
	MyFile << "\n";

	// Writing the cells (cells)
	MyFile << "CELLS " << total_number_of_faces << " " << total_number_of_faces * 4 << "\n";

	for (auto& face : faces)
	{
		MyFile << 3 << " " << face.nodes[0] << " " << face.nodes[1] << " " << face.nodes[2] << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_faces << "\n";

	for (int i = 0; i < total_number_of_faces; i++)
	{
		MyFile << "5\n";
	}

	MyFile << "\n";

	// Writing cell data
	if (number_of_cell_data > 0)
	{
		MyFile << "CELL_DATA " << total_number_of_faces << "\n";
		MyFile << "FIELD FieldData " << number_of_cell_data << "\n";
		MyFile << "\n";

		if (settings.save_face_id == 1)
		{
			MyFile << "FaceID_(#) " << 1 << " " << total_number_of_faces << " int\n";
			for (auto& face : faces)
			{
				MyFile << face.id << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_face_cell_id == 1)
		{
			MyFile << "CellID_(#) " << 1 << " " << total_number_of_faces << " int\n";
			for (auto& face : faces)
			{
				MyFile << face.cell << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_face_area == 1)
		{
			MyFile << "FaceArea_(m2) " << 1 << " " << total_number_of_faces << " double\n";
			for (auto& face : faces)
			{
				MyFile << face.area << "\n";
			}
			MyFile << "\n";
		}
	}

	if (settings.save_face_normal == 1)
	{
		MyFile << "NORMALS FaceNormal_(-) double\n";
		for (auto& face : faces)
		{
			MyFile << face.normal.x << " " << face.normal.y << " " << face.normal.z << "\n";
		}
		MyFile << "\n";
	}

	// Closing the VTK file
	MyFile.close();
}


__host__ void writeCells(CellContainer&		cells,
						 NodeContainer&		nodes,
						 Settings&			settings,
						 int				step,
						 std::string		file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_cells = cells.size();
	int total_number_of_points = nodes.size();
	int number_of_cell_data = 0;

	if (settings.save_cell_id == 1)	number_of_cell_data++;
	if (settings.save_cell_status == 1)	number_of_cell_data++;
	if (settings.save_cell_material_type == 1) number_of_cell_data++;
	if (settings.save_cell_material_property == 1) number_of_cell_data++;
	if (settings.save_cell_volume == 1)	number_of_cell_data++;
	if (settings.save_cell_mass == 1) number_of_cell_data++;

	int number_of_point_data = 0;
	
	if (settings.save_cell_node_id == 1) number_of_point_data++;
	if (settings.save_cell_node_mass == 1) number_of_point_data++;
	if (settings.save_cell_node_force == 1)	number_of_point_data++;
	if (settings.save_cell_node_velocity == 1) number_of_point_data++;
	if (settings.save_cell_node_acceleration == 1) number_of_point_data++;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the POINTS (nodes)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& node : nodes)
	{
		MyFile << node.position.x << " " << node.position.y << " " << node.position.z << "\n";
	}
			
	MyFile << "\n";

	// Writing the CELLS (cells)
	MyFile << "CELLS " << total_number_of_cells << " " << total_number_of_cells * 5 << "\n";

	for (auto& cell : cells)
	{
		MyFile << 4 << " " << cell.nodes[0] << " " << cell.nodes[1] << " " << cell.nodes[2] << " " << cell.nodes[3] << "\n";
	}
		
	MyFile << "\n";

	// Writing CELL_TYPES
	MyFile << "CELL_TYPES " << total_number_of_cells << "\n";

	for (int i = 0; i < total_number_of_cells; i++)
	{
		MyFile << "10\n";
	}

	MyFile << "\n";

	// Writing CELL DATA
	if (number_of_cell_data > 0)
	{
		MyFile << "CELL_DATA " << total_number_of_cells << "\n";
		MyFile << "FIELD FieldData " << number_of_cell_data << "\n";
		MyFile << "\n";

		if (settings.save_cell_id == 1)
		{
			MyFile << "CellID_(#) " << 1 << " " << total_number_of_cells << " int\n";
			for (auto& cell : cells)
			{
				MyFile << cell.id << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_status == 1)
		{
			MyFile << "CellStatus_(-) " << 1 << " " << total_number_of_cells << " int\n";
			for (auto& cell : cells)
			{
				MyFile << cell.status << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_material_type == 1)
		{
			MyFile << "CellMaterialType_(-) " << 1 << " " << total_number_of_cells << " int\n";
			for (auto& cell : cells)
			{
				MyFile << cell.material_type << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_material_property == 1)
		{
			MyFile << "CellMaterialProperty_(-) " << 1 << " " << total_number_of_cells << " int\n";
			for (auto& cell : cells)
			{
				MyFile << cell.material_property << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_volume == 1)
		{
			MyFile << "CellVolume_(m3) " << 1 << " " << total_number_of_cells << " double\n";
			for (auto& cell : cells)
			{
				MyFile << cell.volume << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_mass == 1)
		{
			MyFile << "CellMass_(kg) " << 1 << " " << total_number_of_cells << " double\n";
			for (auto& cell : cells)
			{
				MyFile << cell.mass << "\n";
			}
			MyFile << "\n";
		}
	}

	// Writing POINT DATA
	if (number_of_point_data > 0)
	{
		MyFile << "POINT_DATA " << total_number_of_points << "\n";
		MyFile << "FIELD FieldData " << number_of_point_data << "\n";
		MyFile << "\n";

		if (settings.save_cell_node_id == 1)
		{
			MyFile << "NodeID_(#) " << 1 << " " << total_number_of_points << " int\n";
			for (auto& node : nodes)
			{
				MyFile << node.id << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_node_mass == 1)
		{
			MyFile << "NodalMass_(kg) " << 1 << " " << total_number_of_points << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.mass << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_node_force == 1)
		{
			MyFile << "NodalForce_(N) " << 3 << " " << total_number_of_points << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.force.x << " " << node.force.y << " " << node.force.z << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_node_velocity == 1)
		{
			MyFile << "NodalVelocity_(m/s) " << 3 << " " << total_number_of_points << " double\n";
			for (auto& node : nodes)
			{

				MyFile << node.velocity.x << " " << node.velocity.y << " " << node.velocity.z << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_cell_node_acceleration == 1)
		{
			MyFile << "NodalAcceleration_(m/s2) " << 3 << " " << total_number_of_points << " double\n";
			for (auto& node : nodes)
			{
				MyFile << node.acceleration.x << " " << node.acceleration.y << " " << node.acceleration.z << "\n";
			}
			MyFile << "\n";
		}
	}

	// Closing the VTK file
	MyFile.close();
}



__host__ void writeAxialSprings(AxialSpringContainer		axial_springs,
								IntersectionNodeContainer	intersection_nodes, 
								Settings					settings, 
								int							step,
								std::string					file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_springs = axial_springs.size();
	int total_number_of_points = intersection_nodes.size();
	int number_of_cell_data = 0;
	int number_of_field_data = 0;

	if (settings.save_axial_spring_id == 1) number_of_cell_data++;
	if (settings.save_axial_spring_cell_id == 1) number_of_cell_data++;
	if (settings.save_axial_spring_stiffness == 1) number_of_cell_data++;
	if (settings.save_axial_spring_damping == 1) number_of_cell_data++;
	if (settings.save_axial_spring_length == 1) number_of_cell_data++;

	if (settings.save_axial_spring_spring_force == 1) number_of_field_data++;
	if (settings.save_axial_spring_damping_force == 1) number_of_field_data++;
	if (settings.save_axial_spring_total_force == 1) number_of_field_data++;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& spring : axial_springs)
	{
		double3 position_a = intersection_nodes[spring.nodes[0]].position;
		double3 position_b = intersection_nodes[spring.nodes[1]].position;
		MyFile << position_a.x << " " << position_a.y << " " << position_a.z << "\n";
		MyFile << position_b.x << " " << position_b.y << " " << position_b.z << "\n";
	}
	MyFile << "\n";

	// Writing the cells (cells)
	MyFile << "CELLS " << total_number_of_springs << " " << total_number_of_springs * 3 << "\n";

	for (auto& spring : axial_springs)
	{
		MyFile << 2 << " " << spring.nodes[0] << " " << spring.nodes[1] << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_springs << "\n";

	for (int i = 0; i < total_number_of_springs; i++)
	{
		MyFile << "3\n";
	}

	MyFile << "\n";

	// Writin cell data
	if (number_of_cell_data > 0)
	{
		MyFile << "CELL_DATA " << total_number_of_springs << "\n";
		MyFile << "FIELD FieldData " << number_of_cell_data << "\n";
		MyFile << "\n";

		if (settings.save_axial_spring_id == 1)
		{
			MyFile << "SpringID_(#) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.id << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_cell_id == 1)
		{
			MyFile << "CellID_(#) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.cell << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_stiffness == 1)
		{
			MyFile << "SpringStiffness_(N/m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.stiffness << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_damping == 1)
		{
			MyFile << "SpringDamping_(Ns/m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.damping << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_length == 1)
		{
			MyFile << "SpringLength_(m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.current_length << "\n";
			}
			MyFile << "\n";
		}
	}

	// Writing point data
	if (number_of_field_data > 0)
	{
		MyFile << "POINT_DATA " << total_number_of_points << "\n";
		MyFile << "FIELD FieldData " << number_of_field_data << "\n";
		MyFile << "\n";

		if (settings.save_axial_spring_spring_force == 1)
		{
			MyFile << "SpringForce_(N) " << 3 << " " << total_number_of_points << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.spring_force_node_a.x << " " << spring.spring_force_node_a.y << " " << spring.spring_force_node_a.z << "\n";
				MyFile << spring.spring_force_node_b.x << " " << spring.spring_force_node_b.y << " " << spring.spring_force_node_b.z << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_damping_force == 1)
		{
			MyFile << "DampingForce_(N) " << 3 << " " << total_number_of_points << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.damping_force_node_a.x << " " << spring.damping_force_node_a.y << " " << spring.damping_force_node_a.z << "\n";
				MyFile << spring.damping_force_node_b.x << " " << spring.damping_force_node_b.y << " " << spring.damping_force_node_b.z << "\n";
			}
			MyFile << "\n";
		}
		if (settings.save_axial_spring_total_force == 1)
		{
			MyFile << "TotalForce_(N) " << 3 << " " << total_number_of_points << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.total_force_node_a.x << " " << spring.total_force_node_a.y << " " << spring.total_force_node_a.z << "\n";
				MyFile << spring.total_force_node_b.x << " " << spring.total_force_node_b.y << " " << spring.total_force_node_b.z << "\n";
			}
			MyFile << "\n";
		}
	}

	// Closing the VTK file
	MyFile.close();
}



__host__ void writeRotationalSprings(RotationalSpringContainer	rotational_springs,
											 AxialSpringContainer		axial_springs,
											 IntersectionNodeContainer	intersection_nodes,
											 Settings					settings,
											 int						step,
											 std::string				file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_springs = rotational_springs.size();
	int total_number_of_points = total_number_of_springs * 4;

	int number_of_cell_data = 0;
	int number_of_field_data = 0;

	if (settings.save_rotational_spring_id == 1) number_of_cell_data++;
	if (settings.save_rotational_spring_cell_id == 1) number_of_cell_data++;
	if (settings.save_rotational_spring_stiffness == 1) number_of_cell_data++;
	if (settings.save_rotational_spring_angle == 1) number_of_cell_data++;

	if (settings.save_rotational_spring_force == 1) number_of_field_data++;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_points << " double\n";

	for (auto& rspring : rotational_springs)
	{
		int aspring_a_index = rspring.axial_springs[0];
		int aspring_b_index = rspring.axial_springs[1];

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
	MyFile << "\n";

	// Writing the cells (cells)
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

	// Writing cell data
	if (number_of_cell_data > 0)
	{
		MyFile << "CELL_DATA " << total_number_of_springs << "\n";
		MyFile << "FIELD FieldData " << number_of_cell_data << "\n";
		MyFile << "\n";

		if (settings.save_rotational_spring_id == 1)
		{
			MyFile << "SpringID_(#) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& rspring : rotational_springs)
			{
					MyFile << rspring.id << "\n";			}
			MyFile << "\n";
		}

		if (settings.save_rotational_spring_cell_id == 1)
		{
			MyFile << "CellID_(#) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& rspring : rotational_springs)
			{
				MyFile << rspring.cell << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_rotational_spring_stiffness == 1)
		{
			MyFile << "SpringStiffness_(N/rad) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& rspring : rotational_springs)
			{
				MyFile << rspring.stiffness << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_rotational_spring_angle == 1)
		{
			MyFile << "SpringAngle_(rad) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& rspring : rotational_springs)
			{
				MyFile << rspring.current_angle << "\n";
			}
			MyFile << "\n";
		}
	}

	// Writing point data
	if (number_of_field_data > 0)
	{
		MyFile << "POINT_DATA " << total_number_of_points << "\n";
		MyFile << "FIELD FieldData " << number_of_field_data << "\n";
		MyFile << "\n";

		MyFile << "SpringForce_(N) " << 3 << " " << total_number_of_points << " double\n";
		for (auto& rspring : rotational_springs)
		{
			MyFile << rspring.spring_a_node_a_force.x << " " << rspring.spring_a_node_a_force.y << " " << rspring.spring_a_node_a_force.z << "\n";
			MyFile << rspring.spring_b_node_a_force.x << " " << rspring.spring_b_node_a_force.y << " " << rspring.spring_b_node_a_force.z << "\n";
			MyFile << rspring.spring_a_node_b_force.x << " " << rspring.spring_a_node_b_force.y << " " << rspring.spring_a_node_b_force.z << "\n";
			MyFile << rspring.spring_b_node_b_force.x << " " << rspring.spring_b_node_b_force.y << " " << rspring.spring_b_node_b_force.z << "\n";
		}
		MyFile << "\n";
	}

	// Closing the VTK file
	MyFile.close();
}