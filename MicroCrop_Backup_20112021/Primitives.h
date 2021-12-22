#pragma once

#include <string>
#include <vector>
#include <map>
#include "cudaVectorMath.h"


//    ____           _     _     _                       
//   / ___|    ___  | |_  | |_  (_)  _ __     __ _   ___ 
//   \___ \   / _ \ | __| | __| | | | '_ \   / _` | / __|
//    ___) | |  __/ | |_  | |_  | | | | | | | (_| | \__ \
//   |____/   \___|  \__|  \__| |_| |_| |_|  \__, | |___/
//                                           |___/       


class Settings
{
public:
	std::vector<std::string> MaterialFiles;
	std::vector<std::string> CellFiles;
	std::vector<std::string> InitialConditionFiles;
	std::vector<std::string> BoundaryConditionFiles;
	std::vector<std::string> ExternalForceFiles;
	std::string OutputFolder;


	double adjust_angle_x = 0.0;
	double adjust_angle_y = 0.0;
	double adjust_angle_z = 0.0;

	double start_time = 0.0;
	double end_time = 0.0;
	double timestep = 0.0;
	double save_interval = 0.0;

	int number_of_CPU_threads = 1;
	int simulation_on_GPU = 0;
	int GPU_device = 0;
	int GPU_threads_per_block = 0;
	int GPU_number_of_blocks = 0;

	int save_nodes = 0;
	int save_node_id = 0;
	int save_node_mass = 0;
	int save_node_force = 0;
	int save_node_velocity = 0;
	int save_node_acceleration = 0;

	int save_cells = 0;
	int save_cell_id = 0;
	int save_cell_status = 0;
	int save_cell_material_type = 0;
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
	int save_face_cell_id = 0;
	int save_face_area = 0;
	int save_face_normal = 0;

	int save_axial_springs = 0;
	int save_axial_spring_id = 0;
	int save_axial_spring_cell_id = 0;
	int save_axial_spring_stiffness = 0;
	int save_axial_spring_damping = 0;
	int save_axial_spring_length = 0;
	int save_axial_spring_spring_force = 0;
	int save_axial_spring_damping_force = 0;
	int save_axial_spring_total_force = 0;

	int save_rotational_springs = 0;
	int save_rotational_spring_id = 0;
	int save_rotational_spring_cell_id = 0;
	int save_rotational_spring_stiffness = 0;
	int save_rotational_spring_angle = 0;
	int save_rotational_spring_force = 0;
	


};


//    __  __           _                   _           _     ____                                         _           
//   |  \/  |   __ _  | |_    ___   _ __  (_)   __ _  | |   |  _ \   _ __    ___    _ __     ___   _ __  | |_   _   _ 
//   | |\/| |  / _` | | __|  / _ \ | '__| | |  / _` | | |   | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | | | |
//   | |  | | | (_| | | |_  |  __/ | |    | | | (_| | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | |_| |
//   |_|  |_|  \__,_|  \__|  \___| |_|    |_|  \__,_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__|  \__, |
//                                                                                 |_|                          |___/ 


class MaterialProperty
{
public:
	// ID of the material property
	int id;

	// Material type
	int type = 0;

	// Location
	double3 location = make_double3(0.0, 0.0, 0.0);

	// Density
	double density = 0.0;

	// Anisotropy axes
	double3 axes[3] = { make_double3(0.0, 0.0, 0.0),
					    make_double3(0.0, 0.0, 0.0),
					    make_double3(0.0, 0.0, 0.0) };
	
	// Axial stiffnesses
	double axial_stiffnesses[3] = { 0.0, 0.0, 0.0 };

	// Axial dampings
	double axial_dampings[3] = { 0.0, 0.0, 0.0 };

	// Rotational stiffnesses
	double rotational_stiffnesses[3] = { 0.0, 0.0, 0.0 };

	// Strengths
	double strength[3] = { 0.0, 0.0, 0.0 };
};

typedef std::vector<MaterialProperty> MaterialContainer;



//    _   _               _        
//   | \ | |   ___     __| |   ___ 
//   |  \| |  / _ \   / _` |  / _ \
//   | |\  | | (_) | | (_| | |  __/
//   |_| \_|  \___/   \__,_|  \___|
//                                 


class Node
{
public:
	// ID of the node
	int id;

	// Mass of the node
	double mass = 0.0;

	// Boundaries
	int3 boundaries = make_int3(0, 0, 0);

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

//    ___           _                                       _     _                     _   _               _        
//   |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___ 
//    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \
//    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/
//   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___|
//                                                                                                                   

class IntersectionNode
{
public:
	// ID of the node
	int id;

	// Status of the node
	int status = 0;

	// Master cell ID
	int cell = 0;

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


//    _____                       
//   |  ___|   __ _    ___    ___ 
//   | |_     / _` |  / __|  / _ \
//   |  _|   | (_| | | (__  |  __/
//   |_|      \__,_|  \___|  \___|
//                                


class Face
{
public:
	// ID of the face
	int id;

	// Status of the face
	int status = 0;

	// Master cell ID
	int cell = 0;

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


//     ____          _   _ 
//    / ___|   ___  | | | |
//   | |      / _ \ | | | |
//   | |___  |  __/ | | | |
//    \____|  \___| |_| |_|
//                         


class Cell
{
public:
	// ID of the cell
	int id;

	// Status of the cell
	int status = 0;

	// Material type
	int material_type = 0;

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

	// Circumsphere radius
	double circumsphere_radius = 0.0;

	// Cell volume
	double volume = 0.0;

	// Cell total mass
	double mass = 0.0;
};

typedef std::vector<Cell> CellContainer;


//       _             _           _     ____                   _                 
//      / \    __  __ (_)   __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _ 
//     / _ \   \ \/ / | |  / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |
//    / ___ \   >  <  | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |
//   /_/   \_\ /_/\_\ |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |
//                                              |_|                         |___/ 



class AxialSpring
{
public:
	// ID of the current spring
	int id;

	// Status of the spring
	int status = 0;

	// Master cell ID
	int cell = 0;

	// Dependent node indices
	int nodes[2] = { 0 , 0 };

	// Spring stiffness
	double stiffness = 0.0;

	// Spring damping
	double damping = 0.0;

	// Spring strength
	double strength = 0.0;

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



//    ____            _             _     _                           _     ____                   _                 
//   |  _ \    ___   | |_    __ _  | |_  (_)   ___    _ __     __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _ 
//   | |_) |  / _ \  | __|  / _` | | __| | |  / _ \  | '_ \   / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |
//   |  _ <  | (_) | | |_  | (_| | | |_  | | | (_) | | | | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |
//   |_| \_\  \___/   \__|  \__,_|  \__| |_|  \___/  |_| |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |
//                                                                                 |_|                         |___/ 


class RotationalSpring
{
public:
	// ID of the current spring
	int id;

	// Status of the spring
	int status = 0;

	// Master cell ID
	int cell = 0;

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


//    _____          _                                   _     _____                              
//   | ____| __  __ | |_    ___   _ __   _ __     __ _  | |   |  ___|   ___    _ __    ___    ___ 
//   |  _|   \ \/ / | __|  / _ \ | '__| | '_ \   / _` | | |   | |_     / _ \  | '__|  / __|  / _ \
//   | |___   >  <  | |_  |  __/ | |    | | | | | (_| | | |   |  _|   | (_) | | |    | (__  |  __/
//   |_____| /_/\_\  \__|  \___| |_|    |_| |_|  \__,_| |_|   |_|      \___/  |_|     \___|  \___|
//                                                                                                

class ExternalForce
{
public:
	// ID of the current external force
	int id;

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


//     ____                   _                    _   
//    / ___|   ___    _ __   | |_    __ _    ___  | |_ 
//   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|
//   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_ 
//    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|
//                                                     

class Contact
{
	// ID of the current contact
	int id;

	// Status of the contact
	int status = 0;

	// Index of cell A (node)
	int cell_a = 0;

	// Index of cell B (face)
	int cell_b = 0;

	// Cell A node index
	int node = 0;

	// Intersection node
	IntersectionNode intersection_node;

	// Stiffness
	double stiffness = 0.0;

	// Normal overlap
	double normal_overlap = 0.0;

	// Tangential overlap
	double tangential_overlap = 0.0;

	// Normal force
	double3 normal_force = make_double3(0.0, 0.0, 0.0);

	// Tangential force
	double3 tangential_force = make_double3(0.0, 0.0, 0.0);

	// Total force
	double3 total_force = make_double3(0.0, 0.0, 0.0);

};

typedef std::vector<Contact> ContactContainer;