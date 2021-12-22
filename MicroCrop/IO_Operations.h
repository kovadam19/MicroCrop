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
#include <iostream>
#include <thread>

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Other includes
#include "Primitives.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _                         ____            _           
//   | |       ___     __ _    |  _ \    __ _  | |_    __ _ 
//   | |      / _ \   / _` |   | | | |  / _` | | __|  / _` |
//   | |___  | (_) | | (_| |   | |_| | | (_| | | |_  | (_| |
//   |_____|  \___/   \__, |   |____/   \__,_|  \__|  \__,_|
//                    |___/                                 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes message onto screeen and saves it inot a log file
template <typename T>
inline __host__ void logData(const T	data,
							 Settings&	settings)
{
	// Opening the log file
	std::string full_file_name = settings.OutputFolder + "LogFile.txt";
	std::ofstream MyFile(full_file_name, std::ios_base::app);

	// Writing message to the screen
	std::cout << data << std::endl;

	// Writing data to the log file
	MyFile << data << "\n";

	// Closing the log file
	MyFile.close();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _                           _      ___            _                     _       _____           _       _               
//   | |       ___     __ _    __| |    / _ \   _   _  | |_   _ __    _   _  | |_    |  ___|   ___   | |   __| |   ___   _ __ 
//   | |      / _ \   / _` |  / _` |   | | | | | | | | | __| | '_ \  | | | | | __|   | |_     / _ \  | |  / _` |  / _ \ | '__|
//   | |___  | (_) | | (_| | | (_| |   | |_| | | |_| | | |_  | |_) | | |_| | | |_    |  _|   | (_) | | | | (_| | |  __/ | |   
//   |_____|  \___/   \__,_|  \__,_|    \___/   \__,_|  \__| | .__/   \__,_|  \__|   |_|      \___/  |_|  \__,_|  \___| |_|   
//                                                           |_|                                                              
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Parses the settings file to find the output folder definition
inline __host__ void loadOutputFolder(Settings& settings)
{
	// Open the settings file
	std::ifstream MyFile("0_INPUT/Settings.txt");

	// Create a string for the items
	std::string item;

	// Reads the file until it ends
	while (MyFile >> item)
	{
		// Check if the item is the OUTPUTFOLDER keyword
		if (item == "OUTPUTFOLDER")
		{
			MyFile >> settings.OutputFolder;
			break;
		}
	}

	// Close the file
	MyFile.close();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _                           _     ____           _     _     _                       
//   | |       ___     __ _    __| |   / ___|    ___  | |_  | |_  (_)  _ __     __ _   ___ 
//   | |      / _ \   / _` |  / _` |   \___ \   / _ \ | __| | __| | | | '_ \   / _` | / __|
//   | |___  | (_) | | (_| | | (_| |    ___) | |  __/ | |_  | |_  | | | | | | | (_| | \__ \
//   |_____|  \___/   \__,_|  \__,_|   |____/   \___|  \__|  \__| |_| |_| |_|  \__, | |___/
//                                                                             |___/       
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Loads settings from the settings file
inline __host__ int loadSettings(Settings& settings)
{
	// Open the material peropeties file
	std::ifstream MyFile("0_INPUT/Settings.txt");

	// Create a string for the items
	std::string item;

	// Reads the file until it ends
	while (MyFile >> item)
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//    ___   _                  _          _     _                   ___                                                  _        
		//   / __| (_)  _ __    _  _  | |  __ _  | |_  (_)  ___   _ _      / __|  ___   _ __    _ __   ___   _ _    ___   _ _   | |_   ___
		//   \__ \ | | | '  \  | || | | | / _` | |  _| | | / _ \ | ' \    | (__  / _ \ | '  \  | '_ \ / _ \ | ' \  / -_) | ' \  |  _| (_-<
		//   |___/ |_| |_|_|_|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_|    \___| \___/ |_|_|_| | .__/ \___/ |_||_| \___| |_||_|  \__| /__/
		//                                                                                     |_|                                        
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Check if the item is the NUMBER_OF_COMPONENTS keyword
		if (item == "NUMBER_OF_COMPONENTS")
		{
			// Read in the number of components
			MyFile >> settings.number_of_components;
		}

		// Check if the item is the MATERIALS keyword
		if (item == "MATERIALS")
		{
			// Variable for the number of material files
			int number_of_material_files = 0;

			// Read in the number of materials we have to read
			MyFile >> number_of_material_files;

			// Check if the number of components and the number of material files are equal
			if (settings.number_of_components == number_of_material_files)
			{
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
			else
			{
				std::cout << "WARNING: The number of material files has to be equal to the number of components!" << std::endl;
				return 1;
			}
		}

		// Check if the item is the INTERACTIONS keyword
		if (item == "INTERACTIONS")
		{
			// Variable for the number of interaction files
			int number_of_interaction_files = 0;

			// Read in the number of interactions we have to read
			MyFile >> number_of_interaction_files;

			// Read the interaction peroperties
			for (int i = 0; i < number_of_interaction_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the file name
				MyFile >> file_name;

				// Add it to the settings
				settings.InteractionFiles.push_back(file_name);
			}
		}

		// Check if the item is the CELLS keyword
		if (item == "CELLS")
		{
			// Variable for the number of cell files
			int number_of_cell_files = 0;

			// Read in the number of materials we have to read
			MyFile >> number_of_cell_files;

			// Check if the number of cell files are equal to the number of components
			if (settings.number_of_components == number_of_cell_files)
			{
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
			else
			{
				std::cout << "WARNING: The number of cell files has to be equal to the number of components!" << std::endl;
				return 1;
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

		// Check if the item is the FIXED_VELOCITY keyword
		if (item == "FIXED_VELOCITY")
		{
			// Variable for the number of fixed velocity files
			int number_of_fixed_velocity_files = 0;

			// Read in the number of fixed velocity files we have to read
			MyFile >> number_of_fixed_velocity_files;

			// Read the file names
			for (int i = 0; i < number_of_fixed_velocity_files; i++)
			{
				// Temporary variable for the file name
				std::string file_name;

				// Load the next file name
				MyFile >> file_name;

				// Add it to the settings
				settings.FixedVelocityFiles.push_back(file_name);
			}
		}

		// Check if the item is the OUTPUTFOLDER keyword
		if (item == "OUTPUTFOLDER")	MyFile >> settings.OutputFolder;

		/////////////////////////////////////////////////////////////////////////////////////////////////
		//    ___         _                       ___         _     _     _                    
		//   / __|  ___  | | __ __  ___   _ _    / __|  ___  | |_  | |_  (_)  _ _    __ _   ___
		//   \__ \ / _ \ | | \ V / / -_) | '_|   \__ \ / -_) |  _| |  _| | | | ' \  / _` | (_-<
		//   |___/ \___/ |_|  \_/  \___| |_|     |___/ \___|  \__|  \__| |_| |_||_| \__, | /__/
		//                                                                          |___/      
		////////////////////////////////////////////////////////////////////////////////////////////////

		// Check if the item is the NUMBER_OF_CPU_THREADS keyword
		if (item == "NUMBER_OF_CPU_THREADS") MyFile >> settings.number_of_CPU_threads;

		// Check if the item is the SIMULATION_ON_GPU keyword
		if (item == "SIMULATION_ON_GPU") MyFile >> settings.simulation_on_GPU;

		// Check if the item is the GPU_DEVICE keyword
		if (item == "GPU_DEVICE") MyFile >> settings.GPU_device;

		// Check if the item is the GPU_THREADS_PER_BLOCK keyword
		if (item == "GPU_THREADS_PER_BLOCK") MyFile >> settings.GPU_threads_per_block;

		// Check if the item is the GPU_NUMBER_OF_BLOCKS keyword
		if (item == "GPU_NUMBER_OF_BLOCKS") MyFile >> settings.GPU_number_of_blocks;

		// Check if the item is the START_TIME keyword
		if (item == "START_TIME") MyFile >> settings.start_time;

		// Check if the item is the END_TIME keyword
		if (item == "END_TIME")	MyFile >> settings.end_time;

		// Check if the item is the TIMESTEP keyword
		if (item == "TIMESTEP")	MyFile >> settings.timestep;

		// Check if the item is the GLOBAL_DAMPING keyword
		if (item == "GLOBAL_DAMPING") MyFile >> settings.global_damping;

		// Check if the item is the SAVE_INTERVAL keyword
		if (item == "SAVE_INTERVAL") MyFile >> settings.save_interval;

		///////////////////////////////////////////////////////////////////////////////////////////////////////
		//      _        _     _               _       __  __          _                 _          _      
		//     /_\    __| |   (_)  _  _   ___ | |_    |  \/  |  __ _  | |_   ___   _ _  (_)  __ _  | |  ___
		//    / _ \  / _` |   | | | || | (_-< |  _|   | |\/| | / _` | |  _| / -_) | '_| | | / _` | | | (_-<
		//   /_/ \_\ \__,_|  _/ |  \_,_| /__/  \__|   |_|  |_| \__,_|  \__| \___| |_|   |_| \__,_| |_| /__/
		//                  |__/                                                                           
		//////////////////////////////////////////////////////////////////////////////////////////////////////

		// Check if the item is the ADJUST_ANGLE_X keyword
		if (item == "ADJUST_ANGLE_X") MyFile >> settings.adjust_angle_x;

		// Check if the item is the ADJUST_ANGLE_Y keyword
		if (item == "ADJUST_ANGLE_Y") MyFile >> settings.adjust_angle_y;

		// Check if the item is the ADJUST_ANGLE_Z keyword
		if (item == "ADJUST_ANGLE_Z") MyFile >> settings.adjust_angle_z;

		////////////////////////////////////////////////////////////
		//     ___         _   _     ___                    
		//    / __|  ___  | | | |   / __|  __ _  __ __  ___ 
		//   | (__  / -_) | | | |   \__ \ / _` | \ V / / -_)
		//    \___| \___| |_| |_|   |___/ \__,_|  \_/  \___|
		//                                                  
		///////////////////////////////////////////////////////////

		// Check if the item is the SAVE_CELLS keyword
		if (item == "SAVE_CELLS") MyFile >> settings.save_cells;

		// Check if the item is the SAVE_CELL_ID keyword
		if (item == "SAVE_CELL_ID") MyFile >> settings.save_cell_id;

		// Check if the item is the SAVE_CELL_COMPONENT keyword
		if (item == "SAVE_CELL_COMPONENT") MyFile >> settings.save_cell_component;

		// Check if the item is the SAVE_CELL_STATUS keyword
		if (item == "SAVE_CELL_STATUS")	MyFile >> settings.save_cell_status;

		// Check if the item is the SAVE_CELL_MATERIAL_PROPERTY keyword
		if (item == "SAVE_CELL_MATERIAL_PROPERTY") MyFile >> settings.save_cell_material_property;

		// Check if the item is the SAVE_CELL_VOLUME keyword
		if (item == "SAVE_CELL_VOLUME") MyFile >> settings.save_cell_volume;

		// Check if the item is the SAVE_CELL_MASS keyword
		if (item == "SAVE_CELL_MASS") MyFile >> settings.save_cell_mass;

		// Check if the item is the SAVE_CELL_NODE_ID keyword
		if (item == "SAVE_CELL_NODE_ID") MyFile >> settings.save_cell_node_id;

		// Check if the item is the SAVE_CELL_NODE_MASS keyword
		if (item == "SAVE_CELL_NODE_MASS") MyFile >> settings.save_cell_node_mass;

		// Check if the item is the SAVE_CELL_NODE_FORCE keyword
		if (item == "SAVE_CELL_NODE_FORCE") MyFile >> settings.save_cell_node_force;

		// Check if the item is the SAVE_CELL_NODE_VELOCITY keyword
		if (item == "SAVE_CELL_NODE_VELOCITY") MyFile >> settings.save_cell_node_velocity;

		// Check if the item is the SAVE_CELL_NODE_ACCELERATION keyword
		if (item == "SAVE_CELL_NODE_ACCELERATION") MyFile >> settings.save_cell_node_acceleration;

		/////////////////////////////////////////////////////////////////
		//    ___                       ___                    
		//   | __|  __ _   __   ___    / __|  __ _  __ __  ___ 
		//   | _|  / _` | / _| / -_)   \__ \ / _` | \ V / / -_)
		//   |_|   \__,_| \__| \___|   |___/ \__,_|  \_/  \___|
		//                                                     
		////////////////////////////////////////////////////////////////

		// Check if the item is the SAVE_FACES keyword
		if (item == "SAVE_FACES") MyFile >> settings.save_faces;

		// Check if the item is the SAVE_FACE_ID keyword
		if (item == "SAVE_FACE_ID") MyFile >> settings.save_face_id;

		// Check if the item is the SAVE_FACE_COMPONENT keyword
		if (item == "SAVE_FACE_COMPONENT") MyFile >> settings.save_face_component;

		// Check if the item is the SAVE_FACE_CELL_ID keyword
		if (item == "SAVE_FACE_CELL_ID") MyFile >> settings.save_face_cell_id;

		// Check if the item is the SAVE_FACE_AREA keyword
		if (item == "SAVE_FACE_AREA") MyFile >> settings.save_face_area;

		// Check if the item is the SAVE_FACE_NORMAL keyword
		if (item == "SAVE_FACE_NORMAL") MyFile >> settings.save_face_normal;

		/////////////////////////////////////////////////////////////////
		//    _  _            _           ___                    
		//   | \| |  ___   __| |  ___    / __|  __ _  __ __  ___ 
		//   | .` | / _ \ / _` | / -_)   \__ \ / _` | \ V / / -_)
		//   |_|\_| \___/ \__,_| \___|   |___/ \__,_|  \_/  \___|
		//                                                       
		////////////////////////////////////////////////////////////////

		// Check if the item is the SAVE_NODES keyword
		if (item == "SAVE_NODES") MyFile >> settings.save_nodes;

		// Check if the item is the SAVE_NODE_ID keyword
		if (item == "SAVE_NODE_ID") MyFile >> settings.save_node_id;

		// Check if the item is the SAVE_NODE_COMPONENT keyword
		if (item == "SAVE_NODE_COMPONENT") MyFile >> settings.save_node_component;

		// Check if the item is the SAVE_NODE_MASS keyword
		if (item == "SAVE_NODE_MASS") MyFile >> settings.save_node_mass;

		// Check if the item is the SAVE_NODE_FORCE keyword
		if (item == "SAVE_NODE_FORCE") MyFile >> settings.save_node_force;

		// Check if the item is the SAVE_NODE_VELOCITY keyword
		if (item == "SAVE_NODE_VELOCITY") MyFile >> settings.save_node_velocity;

		// Check if the item is the SAVE_NODE_ACCELERATION keyword
		if (item == "SAVE_NODE_ACCELERATION") MyFile >> settings.save_node_acceleration;

		// Check if the item is the SAVE_NODE_FIXED_VELOCITY keyword
		if (item == "SAVE_NODE_FIXED_VELOCITY") MyFile >> settings.save_node_fixed_velocity;

		// Check if the item is the SAVE_NODE_BOUNDARIES keyword
		if (item == "SAVE_NODE_BOUNDARIES") MyFile >> settings.save_node_boundaries;

		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//      _           _          _     ___                _                   ___                    
		//     /_\   __ __ (_)  __ _  | |   / __|  _ __   _ _  (_)  _ _    __ _    / __|  __ _  __ __  ___ 
		//    / _ \  \ \ / | | / _` | | |   \__ \ | '_ \ | '_| | | | ' \  / _` |   \__ \ / _` | \ V / / -_)
		//   /_/ \_\ /_\_\ |_| \__,_| |_|   |___/ | .__/ |_|   |_| |_||_| \__, |   |___/ \__,_|  \_/  \___|
		//                                        |_|                     |___/                            
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Check if the item is the SAVE_AXIAL_SPRINGS keyword
		if (item == "SAVE_AXIAL_SPRINGS") MyFile >> settings.save_axial_springs;

		// Check if the item is the SAVE_AXIAL_SPRING_ID keyword
		if (item == "SAVE_AXIAL_SPRING_ID") MyFile >> settings.save_axial_spring_id;

		// Check if the item is the SAVE_AXIAL_SPRING_COMPONENT keyword
		if (item == "SAVE_AXIAL_SPRING_COMPONENT") MyFile >> settings.save_axial_spring_component;

		// Check if the item is the SAVE_AXIAL_SPRING_CELL_ID keyword
		if (item == "SAVE_AXIAL_SPRING_CELL_ID") MyFile >> settings.save_axial_spring_cell_id;

		// Check if the item is the SAVE_AXIAL_SPRING_TENSILE_STIFFNESS keyword
		if (item == "SAVE_AXIAL_SPRING_TENSILE_STIFFNESS") MyFile >> settings.save_axial_spring_tensile_stiffness;

		// Check if the item is the SAVE_AXIAL_SPRING_COMPRESSIVE_STIFFNESS keyword
		if (item == "SAVE_AXIAL_SPRING_COMPRESSIVE_STIFFNESS") MyFile >> settings.save_axial_spring_compressive_stiffness;

		// Check if the item is the SAVE_AXIAL_SPRING_TENSILE_DAMPING keyword
		if (item == "SAVE_AXIAL_SPRING_TENSILE_DAMPING") MyFile >> settings.save_axial_spring_tensile_damping;

		// Check if the item is the SAVE_AXIAL_SPRING_COMPRESSIVE_DAMPING keyword
		if (item == "SAVE_AXIAL_SPRING_COMPRESSIVE_DAMPING") MyFile >> settings.save_axial_spring_compressive_damping;

		// Check if the item is the SAVE_AXIAL_SPRING_TENSILE_STRENGTH keyword
		if (item == "SAVE_AXIAL_SPRING_TENSILE_STRENGTH") MyFile >> settings.save_axial_spring_tensile_strength;

		// Check if the item is the SAVE_AXIAL_SPRING_COMPRESSIVE_STRENGTH keyword
		if (item == "SAVE_AXIAL_SPRING_COMPRESSIVE_STRENGTH") MyFile >> settings.save_axial_spring_compressive_strength;

		// Check if the item is the SAVE_AXIAL_SPRING_LOADCASE keyword
		if (item == "SAVE_AXIAL_SPRING_LOADCASE") MyFile >> settings.save_axial_spring_loadcase;

		// Check if the item is the SAVE_AXIAL_SPRING_LENGTH keyword
		if (item == "SAVE_AXIAL_SPRING_LENGTH") MyFile >> settings.save_axial_spring_length;

		// Check if the item is the SAVE_AXIAL_SPRING_SPRING_FORCE keyword
		if (item == "SAVE_AXIAL_SPRING_SPRING_FORCE") MyFile >> settings.save_axial_spring_spring_force;

		// Check if the item is the SAVE_AXIAL_SPRING_DAMPING_FORCE keyword
		if (item == "SAVE_AXIAL_SPRING_DAMPING_FORCE") MyFile >> settings.save_axial_spring_damping_force;

		// Check if the item is the SAVE_AXIAL_SPRING_TOTAL_FORCE keyword
		if (item == "SAVE_AXIAL_SPRING_TOTAL_FORCE") MyFile >> settings.save_axial_spring_total_force;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//    ___         _            _     _                       _     ___                _                   ___                    
		//   | _ \  ___  | |_   __ _  | |_  (_)  ___   _ _    __ _  | |   / __|  _ __   _ _  (_)  _ _    __ _    / __|  __ _  __ __  ___ 
		//   |   / / _ \ |  _| / _` | |  _| | | / _ \ | ' \  / _` | | |   \__ \ | '_ \ | '_| | | | ' \  / _` |   \__ \ / _` | \ V / / -_)
		//   |_|_\ \___/  \__| \__,_|  \__| |_| \___/ |_||_| \__,_| |_|   |___/ | .__/ |_|   |_| |_||_| \__, |   |___/ \__,_|  \_/  \___|
		//                                                                      |_|                     |___/                            
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Check if the item is the SAVE_ROTATIONAL_SPRINGS keyword
		if (item == "SAVE_ROTATIONAL_SPRINGS") MyFile >> settings.save_rotational_springs;

		// Check if the item is the SAVE_ROTATIONAL_SPRING_ID keyword
		if (item == "SAVE_ROTATIONAL_SPRING_ID") MyFile >> settings.save_rotational_spring_id;

		// Check if the item is the SAVE_ROTATIONAL_COMPONENT keyword
		if (item == "SAVE_ROTATIONAL_COMPONENT") MyFile >> settings.save_rotational_spring_component;

		// Check if the item is the SAVE_ROTATIONAL_SPRING_CELL_ID keyword
		if (item == "SAVE_ROTATIONAL_SPRING_CELL_ID") MyFile >> settings.save_rotational_spring_cell_id;

		// Check if the item is the SAVE_ROTATIONAL_SPRING_STIFFNESS keyword
		if (item == "SAVE_ROTATIONAL_SPRING_STIFFNESS") MyFile >> settings.save_rotational_spring_stiffness;

		// Check if the item is the SAVE_ROTATIONAL_SPRING_ANGLE keyword
		if (item == "SAVE_ROTATIONAL_SPRING_ANGLE") MyFile >> settings.save_rotational_spring_angle;

		// Check if the item is the SAVE_ROTATIONAL_SPRING_FORCE keyword
		if (item == "SAVE_ROTATIONAL_SPRING_FORCE") MyFile >> settings.save_rotational_spring_force;

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//     ___                _                 _       ___                            ___                    
		//    / __|  ___   _ _   | |_   __ _   __  | |_    | __|  ___   _ _   __   ___    / __|  __ _  __ __  ___ 
		//   | (__  / _ \ | ' \  |  _| / _` | / _| |  _|   | _|  / _ \ | '_| / _| / -_)   \__ \ / _` | \ V / / -_)
		//    \___| \___/ |_||_|  \__| \__,_| \__|  \__|   |_|   \___/ |_|   \__| \___|   |___/ \__,_|  \_/  \___|
		//                                                                                                        
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// Check if the item is the SAVE_CONTACTS keyword
		if (item == "SAVE_CONTACTS") MyFile >> settings.save_contacts;

		// Check if the item is the SAVE_CONTACT_ID keyword
		if (item == "SAVE_CONTACT_ID") MyFile >> settings.save_contact_id;

		// Check if the item is the SAVE_CONTACT_FACE keyword
		if (item == "SAVE_CONTACT_FACE") MyFile >> settings.save_contact_face;

		// Check if the item is the SAVE_CONTACT_FRICTION keyword
		if (item == "SAVE_CONTACT_FRICTION") MyFile >> settings.save_contact_friction;

		// Check if the item is the SAVE_CONTACT_NORMAL_STIFFNESS keyword
		if (item == "SAVE_CONTACT_NORMAL_STIFFNESS") MyFile >> settings.save_contact_normal_stiffness;

		// Check if the item is the SAVE_CONTACT_TANGENTIAL_STIFFNESS keyword
		if (item == "SAVE_CONTACT_TANGENTIAL_STIFFNESS") MyFile >> settings.save_contact_tangential_stiffness;

		// Check if the item is the SAVE_CONTACT_NORMAL_OVERLAP keyword
		if (item == "SAVE_CONTACT_NORMAL_OVERLAP") MyFile >> settings.save_contact_normal_overlap;

		// Check if the item is the SAVE_CONTACT_TANGENTIAL_OVERLAP keyword
		if (item == "SAVE_CONTACT_TANGENTIAL_OVERLAP") MyFile >> settings.save_contact_tangential_overlap;

		// Check if the item is the SAVE_CONTACT_NORMAL_FORCE keyword
		if (item == "SAVE_CONTACT_NORMAL_FORCE") MyFile >> settings.save_contact_normal_force;

		// Check if the item is the SAVE_CONTACT_TANGENTIAL_FORCE keyword
		if (item == "SAVE_CONTACT_TANGENTIAL_FORCE") MyFile >> settings.save_contact_tangential_force;

		// Check if the item is the SAVE_CONTACT_TOTAL_FORCE keyword
		if (item == "SAVE_CONTACT_TOTAL_FORCE") MyFile >> settings.save_contact_total_force;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//    ___         _                               _     ___                            ___                    
		//   | __| __ __ | |_   ___   _ _   _ _    __ _  | |   | __|  ___   _ _   __   ___    / __|  __ _  __ __  ___ 
		//   | _|  \ \ / |  _| / -_) | '_| | ' \  / _` | | |   | _|  / _ \ | '_| / _| / -_)   \__ \ / _` | \ V / / -_)
		//   |___| /_\_\  \__| \___| |_|   |_||_| \__,_| |_|   |_|   \___/ |_|   \__| \___|   |___/ \__,_|  \_/  \___|
		//                                                                                                            
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Check if the item is the SAVE_EXTERNAL_FORCES keyword
		if (item == "SAVE_EXTERNAL_FORCES") MyFile >> settings.save_external_forces;

		// Check if the item is the SAVE_EXTERNAL_FORCE_ID keyword
		if (item == "SAVE_EXTERNAL_FORCE_ID") MyFile >> settings.save_external_force_id;

		// Check if the item is the SAVE_EXTERNAL_FORCE_TYPE keyword
		if (item == "SAVE_EXTERNAL_FORCE_TYPE") MyFile >> settings.save_external_force_type;

		// Check if the item is the SAVE_EXTERNAL_FORCE_VALUE keyword
		if (item == "SAVE_EXTERNAL_FORCE_VALUE") MyFile >> settings.save_external_force_value;
	}

	// Close the file
	MyFile.close();

	// Return
	return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _                           _     __  __           _                   _           _     ____                                         _     _              
//   | |       ___     __ _    __| |   |  \/  |   __ _  | |_    ___   _ __  (_)   __ _  | |   |  _ \   _ __    ___    _ __     ___   _ __  | |_  (_)   ___   ___ 
//   | |      / _ \   / _` |  / _` |   | |\/| |  / _` | | __|  / _ \ | '__| | |  / _` | | |   | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | |  / _ \ / __|
//   | |___  | (_) | | (_| | | (_| |   | |  | | | (_| | | |_  |  __/ | |    | | | (_| | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | | |  __/ \__ \
//   |_____|  \___/   \__,_|  \__,_|   |_|  |_|  \__,_|  \__|  \___| |_|    |_|  \__,_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__| |_|  \___| |___/
//                                                                                                                   |_|                                         
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Loads crop material properties from TXT file
inline __host__ void loadMaterialProperties(Settings&			settings,
											MaterialContainer&	materials)
{
	// Go through the material file names
	for (int component = 0; component < settings.number_of_components; component++)
	{
		// Create a temporary material container
		MaterialContainer temp_container;

		// Open the material peropeties file
		std::ifstream MyFile(settings.MaterialFiles[component]);

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

				// Reserve space for the container
				temp_container.reserve(number_of_materials);

				// Create the material properties
				for (int i = 0; i < number_of_materials; i++)
				{
					// Create a new material instance with the given properties
					MaterialProperty new_material;

					// Create ID for the new material
					new_material.id = _material_id++;

					// Assign the component to the material
					new_material.component = component;

					// Add the material to the material container
					temp_container.push_back(new_material);
				}
			}

			// Check if the item is the LOCATION keyword
			if (item == "LOCATION")
			{
				// Read the locations
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].location.x;
					MyFile >> temp_container[i].location.y;
					MyFile >> temp_container[i].location.z;
				}
			}

			// Check if the item is the DENSITY keyword
			if (item == "DENSITY")
			{
				// Read the densities
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].density;
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
						MyFile >> temp_container[i].axes[j].x;
						MyFile >> temp_container[i].axes[j].y;
						MyFile >> temp_container[i].axes[j].z;
					}	
				}
			}

			// Check if the item is the ANISOTROPY_TENSILE_STIFFNESS keyword
			if (item == "ANISOTROPY_TENSILE_STIFFNESS")
			{
				// Read the anisotropy tensile stiffnesses
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].tensile_axial_stiffnesses[0];
					MyFile >> temp_container[i].tensile_axial_stiffnesses[1];
					MyFile >> temp_container[i].tensile_axial_stiffnesses[2];
				}
			}

			// Check if the item is the ANISOTROPY_COMPRESSIVE_STIFFNESS keyword
			if (item == "ANISOTROPY_COMPRESSIVE_STIFFNESS")
			{
				// Read the anisotropy compressive stiffnesses
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].compressive_axial_stiffnesses[0];
					MyFile >> temp_container[i].compressive_axial_stiffnesses[1];
					MyFile >> temp_container[i].compressive_axial_stiffnesses[2];
				}
			}


			// Check if the item is the ANISOTROPY_TENSILE_DAMPING keyword
			if (item == "ANISOTROPY_TENSILE_DAMPING")
			{
				// Read the anisotropy tensile dampings
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].tensile_axial_dampings[0];
					MyFile >> temp_container[i].tensile_axial_dampings[1];
					MyFile >> temp_container[i].tensile_axial_dampings[2];
				}
			}

			// Check if the item is the ANISOTROPY_COMPRESSIVE_DAMPING keyword
			if (item == "ANISOTROPY_COMPRESSIVE_DAMPING")
			{
				// Read the anisotropy compressive dampings
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].compressive_axial_dampings[0];
					MyFile >> temp_container[i].compressive_axial_dampings[1];
					MyFile >> temp_container[i].compressive_axial_dampings[2];
				}
			}

			// Check if the item is the ANISOTROPY_ROT_STIFFNESS keyword
			if (item == "ANISOTROPY_ROT_STIFFNESS")
			{
				// Read the anisotropy rotational stiffnesses
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].rotational_stiffnesses[0];
					MyFile >> temp_container[i].rotational_stiffnesses[1];
					MyFile >> temp_container[i].rotational_stiffnesses[2];
				}
			}

			// Check if the item is the ANISOTROPY_SPRING_TENSILE_STRENGTH keyword
			if (item == "ANISOTROPY_SPRING_TENSILE_STRENGTH")
			{
				// Read the anisotropy spring strengths
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].tensile_strength[0];
					MyFile >> temp_container[i].tensile_strength[1];
					MyFile >> temp_container[i].tensile_strength[2];
				}
			}

			// Check if the item is the ANISOTROPY_SPRING_COMPRESSIVE_STRENGTH keyword
			if (item == "ANISOTROPY_SPRING_COMPRESSIVE_STRENGTH")
			{
				// Read the anisotropy spring strengths
				for (int i = 0; i < number_of_materials; i++)
				{
					MyFile >> temp_container[i].compressive_strength[0];
					MyFile >> temp_container[i].compressive_strength[1];
					MyFile >> temp_container[i].compressive_strength[2];
				}
			}
		}

		// Insert the temporary container into the materials
		materials.insert(materials.end(), temp_container.begin(), temp_container.end());

		// Close the file
		MyFile.close();
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _                           _     ___           _                                  _     _                     ____                                         _     _              
//   | |       ___     __ _    __| |   |_ _|  _ __   | |_    ___   _ __    __ _    ___  | |_  (_)   ___    _ __     |  _ \   _ __    ___    _ __     ___   _ __  | |_  (_)   ___   ___ 
//   | |      / _ \   / _` |  / _` |    | |  | '_ \  | __|  / _ \ | '__|  / _` |  / __| | __| | |  / _ \  | '_ \    | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | |  / _ \ / __|
//   | |___  | (_) | | (_| | | (_| |    | |  | | | | | |_  |  __/ | |    | (_| | | (__  | |_  | | | (_) | | | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | | |  __/ \__ \
//   |_____|  \___/   \__,_|  \__,_|   |___| |_| |_|  \__|  \___| |_|     \__,_|  \___|  \__| |_|  \___/  |_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__| |_|  \___| |___/
//                                                                                                                                         |_|                                         
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Loads interaction properties from TXT file
inline __host__ void loadInteractionProperties(Settings&						settings,
											   InteractionPropertyContainer&	interaction_properties)
{
	// Go through the interaction property file names
	for (auto& file_name : settings.InteractionFiles)
	{
		// Open the current file
		std::ifstream MyFile(file_name);

		// Create a string for the items
		std::string item;

		// Variable for the number of interactions
		int number_of_interactions = 0;

		// Reads the file until it ends
		while (MyFile >> item)
		{
			// Check if the item is the GLOBAL keyword
			if (item == "INTERACTION")
			{
				// Read in the number of interactions we have to read
				MyFile >> number_of_interactions;

				// Go through the interactions
				for (int i = 0; i < number_of_interactions; i++)
				{
					// Create a new interaction property
					InteractionProperty new_interaction;

					// Assign an ID to the interaction
					new_interaction.id = _interaction_property_id++;

					// Read the first component
					MyFile >> new_interaction.component_a;

					// Read the second component
					MyFile >> new_interaction.component_b;

					// Read the coefficient of static friction
					MyFile >> new_interaction.coefficient_of_static_priction;

					// Read the normal stiffness
					MyFile >> new_interaction.normal_stiffness;

					// Read the tangential stiffness
					MyFile >> new_interaction.tangential_stiffness;

					// Check if component A is greater than component B
					if (new_interaction.component_a > new_interaction.component_b)
					{
						// Swap component A and B
						int temp = new_interaction.component_a;
						new_interaction.component_a = new_interaction.component_b;
						new_interaction.component_b = temp;	
					}

					// Add the interaction to the container
					interaction_properties.push_back(new_interaction);
				}
			}
		}

		// Close the file
		MyFile.close();
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//    _                           _      ____          _   _       
//   | |       ___     __ _    __| |    / ___|   ___  | | | |  ___ 
//   | |      / _ \   / _` |  / _` |   | |      / _ \ | | | | / __|
//   | |___  | (_) | | (_| | | (_| |   | |___  |  __/ | | | | \__ \
//   |_____|  \___/   \__,_|  \__,_|    \____|  \___| |_| |_| |___/
//                                                                 
/////////////////////////////////////////////////////////////////////////////////////////////////


// Load cells from VTK file
inline __host__ void loadCells(Settings&		settings,
							   NodeContainer&	nodes, 
						       CellContainer&	cells)
{
	// Go through the cell file names
	for (int component = 0; component < settings.number_of_components; component++)
	{
		// Open the VTK file
		std::ifstream VTKfile(settings.CellFiles[component]);

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

						// Assign the component to the cell
						new_cell.component = component;

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

								// Assign the component to the new node
								new_node.component = component;

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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____                                               ___                           _       _____   _   _              
//   |  _ \   _ __    ___     ___    ___   ___   ___    |_ _|  _ __    _ __    _   _  | |_    |  ___| (_) | |   ___   ___ 
//   | |_) | | '__|  / _ \   / __|  / _ \ / __| / __|    | |  | '_ \  | '_ \  | | | | | __|   | |_    | | | |  / _ \ / __|
//   |  __/  | |    | (_) | | (__  |  __/ \__ \ \__ \    | |  | | | | | |_) | | |_| | | |_    |  _|   | | | | |  __/ \__ \
//   |_|     |_|     \___/   \___|  \___| |___/ |___/   |___| |_| |_| | .__/   \__,_|  \__|   |_|     |_| |_|  \___| |___/
//                                                                    |_|                                                 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Processes all the input files
inline __host__ int processInputFiles(Settings&						settings,
									  MaterialContainer&			materials,
									  InteractionPropertyContainer& interaction_properties,
									  NodeContainer&				nodes,
									  CellContainer&				cells)
{
	int error = 0;

	logData("Loading settings...", settings);
	error = loadSettings(settings);
	if (error != 0) 
	{ 
		logData("ERROR: Loading the settings failed!", settings);
		return 1; 
	}

	logData("Loading material properties...", settings);
	loadMaterialProperties(settings, materials);

	logData("Loading interaction properties...", settings);
	loadInteractionProperties(settings, interaction_properties);

	logData("Loading cells...", settings);
	loadCells(settings, nodes, cells);

	return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _               ____          _   _     _   _               _              
//   \ \      / /  _ __  (_) | |_    ___     / ___|   ___  | | | |   | \ | |   ___     __| |   ___   ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \   | |      / _ \ | | | |   |  \| |  / _ \   / _` |  / _ \ / __|
//     \ V  V /   | |    | | | |_  |  __/   | |___  |  __/ | | | |   | |\  | | (_) | | (_| | |  __/ \__ \
//      \_/\_/    |_|    |_|  \__|  \___|    \____|  \___| |_| |_|   |_| \_|  \___/   \__,_|  \___| |___/
//                                                                                                       
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes cell node data into VTK file
inline __host__ void writeCellNodes(NodeContainer	nodes,
									Settings		settings,
									const int		step,
									std::string		file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_nodes = nodes.size();
	int number_of_field_data = 0;

	if (settings.save_node_id == 1) number_of_field_data++;
	if (settings.save_node_component == 1) number_of_field_data++;
	if (settings.save_node_mass == 1) number_of_field_data++;
	if (settings.save_node_force == 1) number_of_field_data++;
	if (settings.save_node_velocity == 1) number_of_field_data++;
	if (settings.save_node_acceleration == 1) number_of_field_data++;
	if (settings.save_node_fixed_velocity == 1) number_of_field_data++;
	if (settings.save_node_boundaries == 1) number_of_field_data++;

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

		if (settings.save_node_component == 1)
		{
			MyFile << "Component_(#) " << 1 << " " << total_number_of_nodes << " int\n";
			for (auto& node : nodes)
			{
				MyFile << node.component << "\n";
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

		if (settings.save_node_fixed_velocity == 1)
		{
			MyFile << "FixedVelocity_(1/0) " << 1 << " " << total_number_of_nodes << " int\n";
			for (auto& node : nodes)
			{
				MyFile << node.apply_fixed_velocity << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_node_boundaries == 1)
		{
			MyFile << "Boundaries_(1/0) " << 3 << " " << total_number_of_nodes << " int\n";
			for (auto& node : nodes)
			{
				MyFile << node.boundaries.x << " " << node.boundaries.y << " " << node.boundaries.z << "\n";
			}
			MyFile << "\n";
		}
	}

	// Closing the VTK file
	MyFile.close();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _               ____          _   _     _____                             
//   \ \      / /  _ __  (_) | |_    ___     / ___|   ___  | | | |   |  ___|   __ _    ___    ___   ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \   | |      / _ \ | | | |   | |_     / _` |  / __|  / _ \ / __|
//     \ V  V /   | |    | | | |_  |  __/   | |___  |  __/ | | | |   |  _|   | (_| | | (__  |  __/ \__ \
//      \_/\_/    |_|    |_|  \__|  \___|    \____|  \___| |_| |_|   |_|      \__,_|  \___|  \___| |___/
//                                                                                                      
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes cell face data into VTK file
inline __host__ void writeCellFaces(FaceContainer	faces,
									NodeContainer	nodes, 
									Settings		settings, 
									const int		step,
									std::string		file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_faces = faces.size();
	int total_number_of_points = nodes.size();

	int number_of_cell_data = 0;

	if (settings.save_face_id) number_of_cell_data++;
	if (settings.save_face_component) number_of_cell_data++;
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

		if (settings.save_face_component == 1)
		{
			MyFile << "ComponentID_(#) " << 1 << " " << total_number_of_faces << " int\n";
			for (auto& face : faces)
			{
				MyFile << face.component << "\n";
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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _               ____          _   _       
//   \ \      / /  _ __  (_) | |_    ___     / ___|   ___  | | | |  ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \   | |      / _ \ | | | | / __|
//     \ V  V /   | |    | | | |_  |  __/   | |___  |  __/ | | | | \__ \
//      \_/\_/    |_|    |_|  \__|  \___|    \____|  \___| |_| |_| |___/
//                                                                      
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes cell data into VTK file
inline __host__ void writeCells(CellContainer&		cells,
								NodeContainer&		nodes,
								Settings&			settings,
								const int			step,
								std::string			file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_cells = cells.size();
	int total_number_of_points = nodes.size();
	int number_of_cell_data = 0;

	if (settings.save_cell_id == 1)	number_of_cell_data++;
	if (settings.save_cell_component == 1)	number_of_cell_data++;
	if (settings.save_cell_status == 1)	number_of_cell_data++;
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

		if (settings.save_cell_component == 1)
		{
			MyFile << "ComponentID_(#) " << 1 << " " << total_number_of_cells << " int\n";
			for (auto& cell : cells)
			{
				MyFile << cell.component << "\n";
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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _                 _             _           _     ____                   _                       
//   \ \      / /  _ __  (_) | |_    ___       / \    __  __ (_)   __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _   ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \     / _ \   \ \/ / | |  / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` | / __|
//     \ V  V /   | |    | | | |_  |  __/    / ___ \   >  <  | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| | \__ \
//      \_/\_/    |_|    |_|  \__|  \___|   /_/   \_\ /_/\_\ |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, | |___/
//                                                                                     |_|                         |___/       
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes axial spring data into VTK file
inline __host__ void writeAxialSprings(AxialSpringContainer			axial_springs,
									   IntersectionNodeContainer	intersection_nodes, 
									   Settings						settings, 
									   const int					step,
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
	if (settings.save_axial_spring_component == 1) number_of_cell_data++;
	if (settings.save_axial_spring_cell_id == 1) number_of_cell_data++;
	if (settings.save_axial_spring_tensile_stiffness == 1) number_of_cell_data++;
	if (settings.save_axial_spring_compressive_stiffness == 1) number_of_cell_data++;
	if (settings.save_axial_spring_tensile_damping == 1) number_of_cell_data++;
	if (settings.save_axial_spring_compressive_damping == 1) number_of_cell_data++;
	if (settings.save_axial_spring_tensile_strength == 1) number_of_cell_data++;
	if (settings.save_axial_spring_compressive_strength == 1) number_of_cell_data++;
	if (settings.save_axial_spring_loadcase == 1) number_of_cell_data++;
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

		if (settings.save_axial_spring_component == 1)
		{
			MyFile << "Component_(#) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.component << "\n";
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

		if (settings.save_axial_spring_tensile_stiffness == 1)
		{
			MyFile << "SpringTensileStiffness_(N/m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.tensile_stiffness << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_compressive_stiffness == 1)
		{
			MyFile << "SpringCompressiveStiffness_(N/m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.compressive_stiffness << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_tensile_damping == 1)
		{
			MyFile << "SpringTensileDamping_(Ns/m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.tensile_damping << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_compressive_damping == 1)
		{
			MyFile << "SpringCompressiveDamping_(Ns/m) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.compressive_damping << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_tensile_strength == 1)
		{
			MyFile << "SpringTensileStrength_(N) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.tensile_strength << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_compressive_strength == 1)
		{
			MyFile << "SpringCompressiveStrength_(N) " << 1 << " " << total_number_of_springs << " double\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.compressive_strength << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_axial_spring_loadcase == 1)
		{
			MyFile << "SpringLoadCase_(0/1/2) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& spring : axial_springs)
			{
				MyFile << spring.loadcase << "\n";
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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _              ____            _             _     _                           _     ____                   _                       
//   \ \      / /  _ __  (_) | |_    ___    |  _ \    ___   | |_    __ _  | |_  (_)   ___    _ __     __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _   ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \   | |_) |  / _ \  | __|  / _` | | __| | |  / _ \  | '_ \   / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` | / __|
//     \ V  V /   | |    | | | |_  |  __/   |  _ <  | (_) | | |_  | (_| | | |_  | | | (_) | | | | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| | \__ \
//      \_/\_/    |_|    |_|  \__|  \___|   |_| \_\  \___/   \__|  \__,_|  \__| |_|  \___/  |_| |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, | |___/
//                                                                                                                        |_|                         |___/       
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes rotational spring data into VTK file
inline __host__ void writeRotationalSprings(RotationalSpringContainer	rotational_springs,
											 AxialSpringContainer		axial_springs,
											 IntersectionNodeContainer	intersection_nodes,
											 Settings					settings,
											 const int					step,
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
	if (settings.save_rotational_spring_component == 1) number_of_cell_data++;
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

		if (settings.save_rotational_spring_component == 1)
		{
			MyFile << "Component_(#) " << 1 << " " << total_number_of_springs << " int\n";
			for (auto& rspring : rotational_springs)
			{
				MyFile << rspring.component << "\n";
			}
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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _               ____                   _                    _         
//   \ \      / /  _ __  (_) | |_    ___     / ___|   ___    _ __   | |_    __ _    ___  | |_   ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \   | |      / _ \  | '_ \  | __|  / _` |  / __| | __| / __|
//     \ V  V /   | |    | | | |_  |  __/   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_  \__ \
//      \_/\_/    |_|    |_|  \__|  \___|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__| |___/
//                                                                                                  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Writes contact data into VTK file
inline __host__ void writeContacts(ContactContainer	contacts,
								   NodeContainer	nodes,
								   Settings			settings,
								   const int		step,
								   std::string		file_name)
{
	// Initial calculations
	int total_number_of_contacts = 0;

	for (auto& contact : contacts)
	{
		if (contact.status == 1)
		{
			total_number_of_contacts++;
		}
	}

	int number_of_field_data = 0;

	if (settings.save_contact_id == 1) number_of_field_data++;
	if (settings.save_contact_face == 1) number_of_field_data++;
	if (settings.save_contact_friction == 1) number_of_field_data++;
	if (settings.save_contact_normal_stiffness == 1) number_of_field_data++;
	if (settings.save_contact_tangential_stiffness == 1) number_of_field_data++;
	if (settings.save_contact_normal_overlap == 1) number_of_field_data++;
	if (settings.save_contact_tangential_overlap == 1) number_of_field_data++;
	if (settings.save_contact_normal_force == 1) number_of_field_data++;
	if (settings.save_contact_tangential_force == 1) number_of_field_data++;
	if (settings.save_contact_total_force == 1) number_of_field_data++;

	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_contacts << " double\n";

	for (int i = 0; i < contacts.size(); i++)
	{
		if (contacts[i].status == 1)
		{
			// Get the node position
			MyFile << nodes[i].position.x << " " << nodes[i].position.y << " " << nodes[i].position.z << "\n";
		}

	}
	MyFile << "\n";

	// Writing the cells (cells)
	MyFile << "CELLS " << total_number_of_contacts << " " << total_number_of_contacts * 2 << "\n";

	for (int i = 0; i < total_number_of_contacts; i++)
	{
		MyFile << 1 << " " << i << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_contacts << "\n";

	for (int i = 0; i < total_number_of_contacts; i++)
	{
		MyFile << "1\n";
	}

	MyFile << "\n";

	// Writing point data
	if (number_of_field_data > 0)
	{
		MyFile << "POINT_DATA " << total_number_of_contacts << "\n";
		MyFile << "FIELD FieldData " << number_of_field_data << "\n";
		MyFile << "\n";

		if (settings.save_contact_id == 1)
		{
			MyFile << "ContactID_(#) " << 1 << " " << total_number_of_contacts << " int\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.id << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_face == 1)
		{
			MyFile << "FaceID_(#) " << 1 << " " << total_number_of_contacts << " int\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.cell_face << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_friction == 1)
		{
			MyFile << "Friction_(-) " << 1 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.coefficient_of_static_friction << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_normal_stiffness == 1)
		{
			MyFile << "NormalStiffness_(N/m) " << 1 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.normal_stiffness << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_tangential_stiffness == 1)
		{
			MyFile << "TangentialStiffness_(N/m) " << 1 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.tangential_stiffness << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_normal_overlap == 1)
		{
			MyFile << "NormalOverlap_(m) " << 1 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.normal_overlap << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_tangential_overlap == 1)
		{
			MyFile << "TangentialOverlap_(m) " << 3 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.tangential_overlap.x << " " << contact.tangential_overlap.y << " " << contact.tangential_overlap.z << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_normal_force == 1)
		{
			MyFile << "NormalForce_(N) " << 3 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.normal_force.x << " " << contact.normal_force.y << " " << contact.normal_force.z << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_tangential_force == 1)
		{
			MyFile << "TangentialForce_(N) " << 3 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.tangential_force.x << " " << contact.tangential_force.y << " " << contact.tangential_force.z << "\n";
				}
			}
			MyFile << "\n";
		}

		if (settings.save_contact_total_force == 1)
		{
			MyFile << "TotalForce_(N) " << 3 << " " << total_number_of_contacts << " double\n";
			for (auto& contact : contacts)
			{
				if (contact.status == 1)
				{
					MyFile << contact.total_force.x << " " << contact.total_force.y << " " << contact.total_force.z << "\n";
				}
			}
			MyFile << "\n";
		}
	}

	// Closing the VTK file
	MyFile.close();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   __        __         _   _              _____          _                                   _     _____                                    
//   \ \      / /  _ __  (_) | |_    ___    | ____| __  __ | |_    ___   _ __   _ __     __ _  | |   |  ___|   ___    _ __    ___    ___   ___ 
//    \ \ /\ / /  | '__| | | | __|  / _ \   |  _|   \ \/ / | __|  / _ \ | '__| | '_ \   / _` | | |   | |_     / _ \  | '__|  / __|  / _ \ / __|
//     \ V  V /   | |    | | | |_  |  __/   | |___   >  <  | |_  |  __/ | |    | | | | | (_| | | |   |  _|   | (_) | | |    | (__  |  __/ \__ \
//      \_/\_/    |_|    |_|  \__|  \___|   |_____| /_/\_\  \__|  \___| |_|    |_| |_|  \__,_| |_|   |_|      \___/  |_|     \___|  \___| |___/
//                                                                                                                                             
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Writes cell node data into VTK file
inline __host__ void writeExternalForces(ExternalForceContainer forces,
										 NodeContainer			nodes,
										 Settings				settings,
										 const int				step,
										 std::string			file_name)
{
	// Opening a VTK file
	std::string full_file_name = settings.OutputFolder + file_name + std::to_string(step) + ".vtk";
	std::ofstream MyFile(full_file_name);

	// Initial calculations
	int total_number_of_forces = forces.size();
	int number_of_field_data = 0;

	if (settings.save_external_force_id == 1) number_of_field_data++;
	if (settings.save_external_force_type == 1) number_of_field_data++;
	if (settings.save_external_force_value == 1) number_of_field_data++;

	// Writing the header
	MyFile << "# vtk DataFile Version 4.2\n";
	MyFile << "vtk output\n";
	MyFile << "ASCII\n";
	MyFile << "DATASET UNSTRUCTURED_GRID\n";
	MyFile << "\n";

	// Writing the nodes (points)
	MyFile << "POINTS " << total_number_of_forces << " double\n";

	for (auto& force : forces)
	{
		// Get the node position
		MyFile << nodes[force.node].position.x << " " << nodes[force.node].position.y << " " << nodes[force.node].position.z << "\n";
	}
	MyFile << "\n";

	// Writing the cells (cells)
	MyFile << "CELLS " << total_number_of_forces << " " << total_number_of_forces * 2 << "\n";

	for (int i = 0; i < total_number_of_forces; i++)
	{
		MyFile << 1 << " " << i << "\n";
	}
	MyFile << "\n";

	// Writing cell types
	MyFile << "CELL_TYPES " << total_number_of_forces << "\n";

	for (int i = 0; i < total_number_of_forces; i++)
	{
		MyFile << "1\n";
	}

	MyFile << "\n";

	// Writing point data
	if (number_of_field_data > 0)
	{
		MyFile << "POINT_DATA " << total_number_of_forces << "\n";
		MyFile << "FIELD FieldData " << number_of_field_data << "\n";
		MyFile << "\n";

		if (settings.save_external_force_id == 1)
		{
			MyFile << "ExternalForceID_(#) " << 1 << " " << total_number_of_forces << " int\n";
			for (auto& force : forces)
			{
				MyFile << force.id << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_external_force_type == 1)
		{
			MyFile << "ExternalForceType_(0/1/2/3/4) " << 1 << " " << total_number_of_forces << " int\n";
			for (auto& force : forces)
			{
				MyFile << force.type << "\n";
			}
			MyFile << "\n";
		}

		if (settings.save_external_force_value == 1)
		{
			MyFile << "ExternalForce_(N) " << 3 << " " << total_number_of_forces << " double\n";
			for (auto& force : forces)
			{
				MyFile << force.force.x << " " << force.force.y << " " << force.force.z << "\n";
			}
			MyFile << "\n";
		}
	}

	// Closing the VTK file
	MyFile.close();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____                                 _       ____            _           
//   | ____| __  __  _ __     ___    _ __  | |_    |  _ \    __ _  | |_    __ _ 
//   |  _|   \ \/ / | '_ \   / _ \  | '__| | __|   | | | |  / _` | | __|  / _` |
//   | |___   >  <  | |_) | | (_) | | |    | |_    | |_| | | (_| | | |_  | (_| |
//   |_____| /_/\_\ | .__/   \___/  |_|     \__|   |____/   \__,_|  \__|  \__,_|
//                  |_|                                                         
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Exports all the requested data by using multiple threads
inline __host__ void exportData(CellContainer&				cells,
								FaceContainer&				faces,
								NodeContainer&				nodes,
								IntersectionNodeContainer&	intersection_nodes,
								AxialSpringContainer&		axial_springs,
								RotationalSpringContainer&	rotational_springs,
								ContactContainer&			contacts,
								ExternalForceContainer&		external_forces,
								Settings&					settings,
								const int					step)
{
	// Creating a container for the threads
	std::vector<std::thread> threads;

	// Creating threads
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

	if (settings.save_contacts == 1)
	{
		threads.push_back(std::thread(writeContacts, contacts, nodes, settings, step, "Contacts_Output_"));
	}

	if (settings.save_external_forces == 1)
	{
		threads.push_back(std::thread(writeExternalForces, external_forces, nodes, settings, step, "ExternalForces_OutPut_"));
	}

	// Joining threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}
