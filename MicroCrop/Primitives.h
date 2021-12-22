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
#include <string>
#include <vector>
#include <chrono>

// Other includes
#include "cudaVectorMath.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____           _               _   _     _                    ___   ____        
//   |  _ \   _ __  (_)  _ __ ___   (_) | |_  (_) __   __   ___    |_ _| |  _ \   ___ 
//   | |_) | | '__| | | | '_ ` _ \  | | | __| | | \ \ / /  / _ \    | |  | | | | / __|
//   |  __/  | |    | | | | | | | | | | | |_  | |  \ V /  |  __/    | |  | |_| | \__ \
//   |_|     |_|    |_| |_| |_| |_| |_|  \__| |_|   \_/    \___|   |___| |____/  |___/
//                                                                                    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static int _material_id = 0;
static int _interaction_property_id = 0;
static int _node_id = 0;
static int _intersection_node_id = 0;
static int _face_id = 0;
static int _cell_id = 0;
static int _axial_spring_id = 0;
static int _rotational_spring_id = 0;
static int _contact_id = 0;
static int _external_force_id = 0;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____           _     _     _                       
//   / ___|    ___  | |_  | |_  (_)  _ __     __ _   ___ 
//   \___ \   / _ \ | __| | __| | | | '_ \   / _` | / __|
//    ___) | |  __/ | |_  | |_  | | | | | | | (_| | \__ \
//   |____/   \___|  \__|  \__| |_| |_| |_|  \__, | |___/
//                                           |___/       
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Settings
{
public:
	// Number of components
	int number_of_components = 0;

	// Input files
	std::vector<std::string> MaterialFiles;
	std::vector<std::string> InteractionFiles;
	std::vector<std::string> CellFiles;
	std::vector<std::string> InitialConditionFiles;
	std::vector<std::string> BoundaryConditionFiles;
	std::vector<std::string> ExternalForceFiles;
	std::vector<std::string> FixedVelocityFiles;
	
	// Output folder
	std::string OutputFolder;

	// Adjust angles for the materials
	double adjust_angle_x = 0.0;
	double adjust_angle_y = 0.0;
	double adjust_angle_z = 0.0;

	// Solver settings
	double start_time = 0.0;
	double end_time = 0.0;
	double timestep = 0.0;
	double global_damping = 0.0;
	int number_of_CPU_threads = 0;
	int simulation_on_GPU = 0;
	int GPU_device = 0;
	int GPU_threads_per_block = 0;
	int GPU_number_of_blocks = 0;

	// Saveing settings
	double save_interval = 0.0;

	int save_nodes = 0;
	int save_node_id = 0;
	int save_node_component = 0;
	int save_node_mass = 0;
	int save_node_force = 0;
	int save_node_velocity = 0;
	int save_node_acceleration = 0;
	int save_node_fixed_velocity = 0;
	int save_node_boundaries = 0;

	int save_cells = 0;
	int save_cell_id = 0;
	int save_cell_component = 0;
	int save_cell_status = 0;
	int save_cell_material_property = 0;
	int save_cell_volume = 0;
	int save_cell_mass = 0;
	int save_cell_node_id = 0;
	int save_cell_node_mass = 0;
	int save_cell_node_force = 0;
	int save_cell_node_velocity = 0;
	int save_cell_node_acceleration = 0;

	int save_faces = 0;
	int save_face_id = 0;
	int save_face_component = 0;
	int save_face_cell_id = 0;
	int save_face_area = 0;
	int save_face_normal = 0;

	int save_axial_springs = 0;
	int save_axial_spring_id = 0;
	int save_axial_spring_component = 0;
	int save_axial_spring_cell_id = 0;
	int save_axial_spring_tensile_stiffness = 0;
	int save_axial_spring_compressive_stiffness = 0;
	int save_axial_spring_tensile_damping = 0;
	int save_axial_spring_compressive_damping = 0;
	int save_axial_spring_tensile_strength = 0;
	int save_axial_spring_compressive_strength = 0;
	int save_axial_spring_loadcase = 0;
	int save_axial_spring_length = 0;
	int save_axial_spring_spring_force = 0;
	int save_axial_spring_damping_force = 0;
	int save_axial_spring_total_force = 0;

	int save_rotational_springs = 0;
	int save_rotational_spring_id = 0;
	int save_rotational_spring_component = 0;
	int save_rotational_spring_cell_id = 0;
	int save_rotational_spring_stiffness = 0;
	int save_rotational_spring_angle = 0;
	int save_rotational_spring_force = 0;

	int save_contacts = 0;
	int save_contact_id = 0;
	int save_contact_face = 0;
	int save_contact_friction = 0;
	int save_contact_normal_stiffness = 0;
	int save_contact_tangential_stiffness = 0;
	int save_contact_normal_overlap = 0;
	int save_contact_tangential_overlap = 0;
	int save_contact_normal_force = 0;
	int save_contact_tangential_force = 0;
	int save_contact_total_force = 0;

	int save_external_forces = 0;
	int save_external_force_id = 0;
	int save_external_force_type = 0;
	int save_external_force_value = 0;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    __  __           _                   _           _     ____                                         _           
//   |  \/  |   __ _  | |_    ___   _ __  (_)   __ _  | |   |  _ \   _ __    ___    _ __     ___   _ __  | |_   _   _ 
//   | |\/| |  / _` | | __|  / _ \ | '__| | |  / _` | | |   | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | | | |
//   | |  | | | (_| | | |_  |  __/ | |    | | | (_| | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | |_| |
//   |_|  |_|  \__,_|  \__|  \___| |_|    |_|  \__,_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__|  \__, |
//                                                                                 |_|                          |___/ 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class MaterialProperty
{
public:
	// ID of the material property
	int id;

	// Component
	int component = 0;

	// Location
	double3 location = make_double3(0.0, 0.0, 0.0);

	// Density
	double density = 0.0;

	// Anisotropy axes
	double3 axes[3] = { make_double3(0.0, 0.0, 0.0),
					    make_double3(0.0, 0.0, 0.0),
					    make_double3(0.0, 0.0, 0.0) };
	
	// Tensile axial stiffnesses
	double tensile_axial_stiffnesses[3] = { 0.0, 0.0, 0.0 };

	// Compressive axial stiffnesses
	double compressive_axial_stiffnesses[3] = { 0.0, 0.0, 0.0 };

	// Tensile axial dampings
	double tensile_axial_dampings[3] = { 0.0, 0.0, 0.0 };

	// Compressive axial dampings
	double compressive_axial_dampings[3] = { 0.0, 0.0, 0.0 };

	// Rotational stiffnesses
	double rotational_stiffnesses[3] = { 0.0, 0.0, 0.0 };

	// Tensile strengths
	double tensile_strength[3] = { 0.0, 0.0, 0.0 };

	// Compressive strengths
	double compressive_strength[3] = { 0.0, 0.0, 0.0 };
};

typedef std::vector<MaterialProperty> MaterialContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _                                  _     _                     ____                                         _           
//   |_ _|  _ __   | |_    ___   _ __    __ _    ___  | |_  (_)   ___    _ __     |  _ \   _ __    ___    _ __     ___   _ __  | |_   _   _ 
//    | |  | '_ \  | __|  / _ \ | '__|  / _` |  / __| | __| | |  / _ \  | '_ \    | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | | | |
//    | |  | | | | | |_  |  __/ | |    | (_| | | (__  | |_  | | | (_) | | | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | |_| |
//   |___| |_| |_|  \__|  \___| |_|     \__,_|  \___|  \__| |_|  \___/  |_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__|  \__, |
//                                                                                                       |_|                          |___/ 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class InteractionProperty
{
public:
	// ID for the interaction property
	int id = 0;

	// Component A
	int component_a = 0;

	// Component B
	int component_b = 0;

	// Coefficient of static friction
	double coefficient_of_static_priction = 0.0;

	// Normal stiffness
	double normal_stiffness = 0.0;

	// Tangential stiffness
	double tangential_stiffness = 0.0;
};

typedef std::vector<InteractionProperty> InteractionPropertyContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _        
//   | \ | |   ___     __| |   ___ 
//   |  \| |  / _ \   / _` |  / _ \
//   | |\  | | (_) | | (_| | |  __/
//   |_| \_|  \___/   \__,_|  \___|
//                                 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Node
{
public:
	// ID of the node
	int id;

	// Component
	int component = 0;

	// Mass of the node
	double mass = 0.0;

	// Boundaries
	int3 boundaries = make_int3(0, 0, 0);

	// Apply fixed velocity
	int apply_fixed_velocity = 0;

	// Fixed velocity
	double3 fixed_velocity = make_double3(0.0, 0.0, 0.0);

	// Translational force on the node
	double3 force = make_double3(0.0, 0.0, 0.0);

	// Translational acceleration of the node
	double3 acceleration = make_double3(0.0, 0.0, 0.0);

	// Translational velocity of the node
	double3 velocity = make_double3(0.0, 0.0, 0.0);

	// Position of the node
	double3 position = make_double3(0.0, 0.0, 0.0);
};

typedef std::vector<Node> NodeContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _                                       _     _                     _   _               _        
//   |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___ 
//    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \
//    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/
//   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___|
//                                                                                                                   
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class IntersectionNode
{
public:
	// ID of the node
	int id;

	// Component
	int component = 0;

	// Master cell ID
	int cell = 0;

	// Status of the node
	int status = 0;

	// Independent node indices
	int nodes[3] = { 0, 0, 0 };

	// Shape function coefficients for the independent nodes
	double coefficients[3] = { 0.0, 0.0, 0.0 };

	// Position of the node
	double3 position = make_double3(0.0, 0.0, 0.0);

	// Translational velocity of the node
	double3 velocity = make_double3(0.0, 0.0, 0.0);
};

typedef std::vector<IntersectionNode> IntersectionNodeContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____                       
//   |  ___|   __ _    ___    ___ 
//   | |_     / _` |  / __|  / _ \
//   |  _|   | (_| | | (__  |  __/
//   |_|      \__,_|  \___|  \___|
//                                
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Face
{
public:
	// ID of the face
	int id;

	// Master cell ID
	int cell = 0;

	// Component
	int component = 0;

	// Status of the face
	int status = 0;

	// Indices of nodes that define the face
	// Nodes 0-1-3 define the triangle
	// Node 4 defines the normal direction
	int nodes[4] = { 0, 0, 0, 0 };

	// Center point of the face
	double3 barycenter = make_double3(0.0, 0.0, 0.0);

	// Face normal
	double3 normal = make_double3(0.0, 0.0, 0.0);

	// Distance from node D
	double distance = 0.0;

	// Face area
	double area = 0.0;
};

typedef std::vector<Face> FaceContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____          _   _ 
//    / ___|   ___  | | | |
//   | |      / _ \ | | | |
//   | |___  |  __/ | | | |
//    \____|  \___| |_| |_|
//                         
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Cell
{
public:
	// ID of the cell
	int id;

	// Component
	int component = 0;

	// Status of the cell
	int status = 0;

	// Material property index
	int material_property = 0;

	// Node indices
	int nodes[4] = { 0, 0, 0, 0 };

	// Face indices
	int faces[4] = { 0, 0, 0, 0 };

	// Intersection node indices
	int intersections[6] = { 0, 0, 0, 0, 0, 0 };

	// Axial spring indices
	int axial_springs[3] = { 0, 0, 0 };

	// Rotational spring indices
	int rotational_springs[3] = { 0, 0, 0 };

	// Center point of the cell
	double3 barycenter = make_double3(0.0, 0.0, 0.0);

	// Cell volume
	double volume = 0.0;

	// Cell total mass
	double mass = 0.0;
};

typedef std::vector<Cell> CellContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _             _           _     ____                   _                 
//      / \    __  __ (_)   __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _ 
//     / _ \   \ \/ / | |  / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |
//    / ___ \   >  <  | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |
//   /_/   \_\ /_/\_\ |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |
//                                              |_|                         |___/ 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class AxialSpring
{
public:
	// ID of the current spring
	int id;

	// Component
	int component = 0;

	// Master cell ID
	int cell = 0;

	// Status of the spring
	int status = 0;

	// Dependent node indices
	int nodes[2] = { 0 , 0 };

	// Load case (0 - Not loaded // 1 - Tension // 2 - Compression)
	int loadcase = 0;

	// Tensile spring stiffness
	double tensile_stiffness = 0.0;

	// Compressive spring stiffness
	double compressive_stiffness = 0.0;

	// Tensile spring damping
	double tensile_damping = 0.0;

	// Compressive spring damping
	double compressive_damping = 0.0;

	// Tensile spring strength
	double tensile_strength = 0.0;

	// Compressive spring strength
	double compressive_strength = 0.0;

	// Initial length
	double initial_length = 0.0;

	// Current length
	double current_length = 0.0;

	// Spring forces
	double3 spring_force_node_a = make_double3(0.0, 0.0, 0.0);
	double3 spring_force_node_b = make_double3(0.0, 0.0, 0.0);

	// Damping forces
	double3 damping_force_node_a = make_double3(0.0, 0.0, 0.0);
	double3 damping_force_node_b = make_double3(0.0, 0.0, 0.0);

	// Total forces
	double3 total_force_node_a = make_double3(0.0, 0.0, 0.0);
	double3 total_force_node_b = make_double3(0.0, 0.0, 0.0);
};

typedef std::vector<AxialSpring> AxialSpringContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____            _             _     _                           _     ____                   _                 
//   |  _ \    ___   | |_    __ _  | |_  (_)   ___    _ __     __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _ 
//   | |_) |  / _ \  | __|  / _` | | __| | |  / _ \  | '_ \   / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |
//   |  _ <  | (_) | | |_  | (_| | | |_  | | | (_) | | | | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |
//   |_| \_\  \___/   \__|  \__,_|  \__| |_|  \___/  |_| |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |
//                                                                                 |_|                         |___/ 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class RotationalSpring
{
public:
	// ID of the current spring
	int id;

	// Component
	int component = 0;

	// Master cell ID
	int cell = 0;

	// Status of the spring
	int status = 0;

	// Dependent axial spring indices
	int axial_springs[2] = { 0 , 0 };

	// Spring stiffness
	double stiffness = 0.0;

	// Initial angle
	double initial_angle = 0.0;

	// Current angle
	double current_angle = 0.0;

	// Spring forces
	double3 spring_a_node_a_force = make_double3(0.0, 0.0, 0.0);
	double3 spring_a_node_b_force = make_double3(0.0, 0.0, 0.0);
	double3 spring_b_node_a_force = make_double3(0.0, 0.0, 0.0);
	double3 spring_b_node_b_force = make_double3(0.0, 0.0, 0.0);
};

typedef std::vector<RotationalSpring> RotationalSpringContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____          _                                   _     _____                              
//   | ____| __  __ | |_    ___   _ __   _ __     __ _  | |   |  ___|   ___    _ __    ___    ___ 
//   |  _|   \ \/ / | __|  / _ \ | '__| | '_ \   / _` | | |   | |_     / _ \  | '__|  / __|  / _ \
//   | |___   >  <  | |_  |  __/ | |    | | | | | (_| | | |   |  _|   | (_) | | |    | (__  |  __/
//   |_____| /_/\_\  \__|  \___| |_|    |_| |_|  \__,_| |_|   |_|      \___/  |_|     \___|  \___|
//                                                                                                
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class ExternalForce
{
public:
	// ID of the current external force
	int id;

	// Type (0 - gravity; 1 - global; 2 - component; 3 - plane; 4 - local)
	int type = 0;

	// Node index
	int node = 0;

	// Start time
	double start_time = 0.0;

	// Duration
	double duration = 0.0;

	// Force
	double3 force = make_double3(0.0, 0.0, 0.0);
};

typedef std::vector<ExternalForce> ExternalForceContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____                   _                    _   
//    / ___|   ___    _ __   | |_    __ _    ___  | |_ 
//   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|
//   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_ 
//    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|
//                                                     
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Contact
{
public:
	// ID of the current contact
	int id;

	// Status of the contact
	int status = 0;

	// Index of cell face
	int cell_face = 0;

	// Static friction
	double coefficient_of_static_friction = 0.0;

	// Normal stiffness
	double normal_stiffness = 0.0;

	// Tangential stiffness
	double tangential_stiffness = 0.0;

	// Normal overlap
	double normal_overlap = 0.0;

	// Tangential overlap
	double3 tangential_overlap = make_double3(0.0, 0.0, 0.0);

	// Normal force
	double3 normal_force = make_double3(0.0, 0.0, 0.0);

	// Tangential force
	double3 tangential_force = make_double3(0.0, 0.0, 0.0);

	// Total force
	double3 total_force = make_double3(0.0, 0.0, 0.0);

};

typedef std::vector<Contact> ContactContainer;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____   _                           
//   |_   _| (_)  _ __ ___     ___   _ __ 
//     | |   | | | '_ ` _ \   / _ \ | '__|
//     | |   | | | | | | | | |  __/ | |   
//     |_|   |_| |_| |_| |_|  \___| |_|   
//                                        
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Timer
{
private:
	std::chrono::steady_clock::time_point start;

public:
	inline __host__ void startTimer()
	{
		start = std::chrono::high_resolution_clock::now();
	}

	inline __host__ auto getDuration()
	{
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		return duration.count();
	}
};
