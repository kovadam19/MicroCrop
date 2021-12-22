#include "Simulation.h"


__host__ void processInputFiles(Settings&				settings,
								MaterialContainer&		materials,
								NodeContainer&			nodes,
								CellContainer&			cells)
{
	std::cout << "Loading settings..." << std::endl;
	loadSettings(settings);

	std::cout << "Loading material properties..." << std::endl;
	loadMaterialProperties(settings, materials);

	std::cout << "Loading cells..." << std::endl;
	loadCells(settings, nodes, cells);
}

__host__ void initializeSimulation(CellContainer& cells,
									FaceContainer& faces,
								   NodeContainer& nodes,
									IntersectionNodeContainer& intersection_nodes,
									AxialSpringContainer& axial_springs,
									RotationalSpringContainer& rotational_springs,
									ExternalForceContainer& external_forces,
									ContactContainer& contacts,
									MaterialContainer& materials,
									Settings& settings)
{
	std::cout << "Initializing cells..." << std::endl;
	initializeCells(&cells[0],
						cells.size(),
						&nodes[0],
						&materials[0],
						materials.size(),
						settings);

	std::cout << "Creating cell faces..." << std::endl;
	createFaces(faces,
						cells,
						nodes);

	updateFacesCPU(&faces[0],
						   faces.size(),
						   &nodes[0],
						   settings.number_of_CPU_threads);

	std::cout << "Creating intersection nodes..." << std::endl;
	createIntersectionNodes(intersection_nodes,
									nodes,
									faces, 
									cells, 
									materials);

	updateIntersectionNodesCPU(&intersection_nodes[0],
									   intersection_nodes.size(),
									   &nodes[0],
									   settings.number_of_CPU_threads);

	std::cout << "Creating axial springs..." << std::endl;
	createAxialSprings(axial_springs,
					   cells, 
					   intersection_nodes, 
					   materials);
	
	std::cout << "Creating rotational springs..." << std::endl;
	createRotationalSprings(rotational_springs, 
							cells, 
							axial_springs, 
							intersection_nodes, 
							materials);

	std::cout << "Applying initial conditions..." << std::endl;
	applyInitialConditions(nodes,
						   settings);

	std::cout << "Applying boundary conditions..." << std::endl;
	applyBoundaryConditions(nodes,
						    settings);

	std::cout << "Creating external forces..." << std::endl;
	createExternalForces(external_forces,
						 nodes,
						 settings);

	//std::cout << "Initializing the contact container..." << std::endl;
	//initializeContacts(contacts,
	//				   cells);

}


__host__ void checkSimulation(CellContainer& cells,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	MaterialContainer& materials,
	Settings& settings)
{
	// Variable for the number of adjusted materials
	int number_adjusted_materials = 0;

	// Adjusted materials
	std::vector<int> adjusted_materials;

	// Finding corrupt rotational springs
	for (auto& rspring : rotational_springs)
	{
		// Check if the initial angle of the spring is not 90 deg
		if (rspring.initial_angle <= 1.57 || rspring.initial_angle >= 1.58)
		{
			// We have to rotate the anisotropy axes by a bit around each axis and recalculate the intersection points, axial and rotational springs
			// Get the cell index
			int cell_index = rspring.cell;

			// Get the material index
			int material_index = cells[cell_index].material_property;

			// Check if the material is already adjusted
			bool is_adjusted = false;
			for (auto& adjusted_material : adjusted_materials)
			{
				if (adjusted_material == material_index)
				{
					is_adjusted = true;
				}
			}

			if (!is_adjusted)
			{
				// Add it to the adjusted materials
				adjusted_materials.push_back(material_index);

				// Increase the number of adjusted cells
				number_adjusted_materials++;

				// Rotation matrices
				double angle_x = settings.adjust_angle_x;
				double3 Rx_0 = make_double3(1.0, 0.0, 0.0);
				double3 Rx_1 = make_double3(0.0, cos(angle_x), -sin(angle_x));
				double3 Rx_2 = make_double3(0.0, sin(angle_x), cos(angle_x));

				double angle_y = settings.adjust_angle_y;
				double3 Ry_0 = make_double3(cos(angle_y), 0.0, sin(angle_y));
				double3 Ry_1 = make_double3(0.0, 1.0, 0.0);
				double3 Ry_2 = make_double3(-sin(angle_y), 0.0, cos(angle_y));

				double angle_z = settings.adjust_angle_z;
				double3 Rz_0 = make_double3(cos(angle_z), -sin(angle_z), 0.0);
				double3 Rz_1 = make_double3(sin(angle_z), cos(angle_z), 0.0);
				double3 Rz_2 = make_double3(0.0, 0.0, 1.0);

				// Go through the anisotropy axes
				for (int j = 0; j < 3; j++)
				{
					// Get the current axis
					double3 axis = materials[material_index].axes[j];

					// Rotate the axis around X
					axis.x = dot(axis, Rx_0);
					axis.y = dot(axis, Rx_1);
					axis.z = dot(axis, Rx_2);

					// Rotate the axis around Y
					axis.x = dot(axis, Ry_0);
					axis.y = dot(axis, Ry_1);
					axis.z = dot(axis, Ry_2);

					// Rotate the axis around Z
					axis.x = dot(axis, Rz_0);
					axis.y = dot(axis, Rz_1);
					axis.z = dot(axis, Rz_2);

					// Set the new axis
					materials[material_index].axes[j] = axis;
				}
			}
		}
	}

	std::cout << "Number of adjusted materials: " << number_adjusted_materials << std::endl;

	intersection_nodes.clear();
	axial_springs.clear();
	rotational_springs.clear();

	std::cout << "Creating adjusted intersection nodes..." << std::endl;
	createIntersectionNodes(intersection_nodes,
		nodes,
		faces,
		cells,
		materials);

	updateIntersectionNodesCPU(&intersection_nodes[0],
		intersection_nodes.size(),
		&nodes[0],
		settings.number_of_CPU_threads);

	std::cout << "Creating adjusted axial springs..." << std::endl;
	createAxialSprings(axial_springs,
		cells,
		intersection_nodes,
		materials);

	std::cout << "Creating adjusted rotational springs..." << std::endl;
	createRotationalSprings(rotational_springs,
		cells,
		axial_springs,
		intersection_nodes,
		materials);
}

__host__ void runSimulationCPU(CellContainer& cells,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	Settings& settings)
{
	//auto start = std::chrono::high_resolution_clock::now();
	//exportSimulationData(cells,
	//	faces,
	//	nodes,
	//	intersection_nodes,
	//	axial_springs,
	//	rotational_springs,
	//	settings,
	//	simulation_step);
	//auto stop = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	//std::cout << "Execution time of the initial save in seconds: " << duration.count() << std::endl;

	//start = std::chrono::high_resolution_clock::now();

	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

	double simulation_time = settings.start_time;
	double simulation_end_time = settings.end_time;
	int save_interval = int(settings.save_interval / settings.timestep);
	int step_counter = 0;
	int export_counter = 0;

	while (simulation_time <= simulation_end_time)
	{

		if (step_counter == save_interval)
		{
			if (simulation_time != settings.start_time)
			{
				stop = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
				std::cout << "Execution time of " << save_interval << " iterations in seconds: " << duration.count() << std::endl;
			}



			std::cout << "Simulation is " << (simulation_time / simulation_end_time) * 100 << " % completed." << std::endl;

			start = std::chrono::high_resolution_clock::now();
			std::cout << "Saving at time " << simulation_time << " second." << std::endl;
			exportSimulationData(cells,
								 faces,
								 nodes,
								 intersection_nodes,
								 axial_springs,
								 rotational_springs,
								 settings,
								 export_counter);
			
			
			export_counter++;
			step_counter = 0;

			stop = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
			std::cout << "Execution time of the save in seconds: " << duration.count() << std::endl;

			start = std::chrono::high_resolution_clock::now();
		}

		//auto start = std::chrono::high_resolution_clock::now();


		resetNodesCPU(&nodes[0],
					  nodes.size(),
					  settings.number_of_CPU_threads);

		updateAxialSpringsCPU(&axial_springs[0],
							  axial_springs.size(),
							  &intersection_nodes[0],
							  settings.number_of_CPU_threads);

		applyAxialSpringForces(&axial_springs[0], 
							   axial_springs.size(), 
							   &intersection_nodes[0], 
							   &nodes[0]);

		updateRotationalSpringsCPU(&rotational_springs[0], 
								   rotational_springs.size(), 
								   &axial_springs[0], 
								   &intersection_nodes[0], 
								   settings.number_of_CPU_threads);

		applyRotationalSpringForces(&rotational_springs[0], 
									rotational_springs.size(), 
									&axial_springs[0], 
									&intersection_nodes[0], 
									&nodes[0]);

		applyExternalForces(&external_forces[0],
							external_forces.size(),
							&nodes[0], 
							simulation_time);

		updateNodesCPU(&nodes[0], 
						nodes.size(), 
						settings.number_of_CPU_threads, 
						settings.timestep);

		updateIntersectionNodesCPU(&intersection_nodes[0], 
										   intersection_nodes.size(),
										   &nodes[0], 
										   settings.number_of_CPU_threads);

		updateFacesCPU(&faces[0], 
						faces.size(), 
						&nodes[0], 
						settings.number_of_CPU_threads);


		checkCellDamage(&cells[0],
			cells.size(),
			&faces[0],
			&intersection_nodes[0],
			&axial_springs[0],
			&rotational_springs[0]);


		updateCellsCPU(&cells[0], 
						cells.size(), 
						&nodes[0],
						&intersection_nodes[0],
						&faces[0],
						&axial_springs[0],
						&rotational_springs[0],
						settings.number_of_CPU_threads);

		step_counter++;
		simulation_time += settings.timestep;
		//auto stop = std::chrono::high_resolution_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		//std::cout << "Execution time of one iteration in milliseconds: " << duration.count() << std::endl;
	}
	//stop = std::chrono::high_resolution_clock::now();
	//duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	//std::cout << "Execution time of the solver save in seconds: " << duration.count() << std::endl;

	//start = std::chrono::high_resolution_clock::now();
	exportSimulationData(cells,
						 faces,
						 nodes,
						 intersection_nodes,
						 axial_springs,
						 rotational_springs,
						 settings,
						 export_counter);
	//stop = std::chrono::high_resolution_clock::now();
	//duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	//std::cout << "Execution time of the final save in seconds: " << duration.count() << std::endl;
}



__host__ void exportSimulationData(CellContainer& cells,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	Settings& settings,
	int step)
{
	std::vector<std::thread> threads;

	if (settings.save_nodes == 1)
	{
		threads.push_back(std::thread(writeCellNodes, nodes, settings, step, "Node_Output_"));
	}

	if (settings.save_cells == 1)
	{
		threads.push_back(std::thread(writeCells, cells, nodes, settings, step, "Cell_Output_"));
	}

	if (settings.save_faces == 1)
	{
		threads.push_back(std::thread(writeCellFaces, faces, nodes, settings, step, "Face_Output_"));
	}
	
	if (settings.save_axial_springs == 1)
	{
		threads.push_back(std::thread(writeAxialSprings, axial_springs, intersection_nodes, settings, step, "AxialSprings_Output_"));
	}

	if (settings.save_rotational_springs == 1)
	{
		threads.push_back(std::thread(writeRotationalSprings, rotational_springs, axial_springs, intersection_nodes, settings, step, "RotationalSprings_Output_"));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}