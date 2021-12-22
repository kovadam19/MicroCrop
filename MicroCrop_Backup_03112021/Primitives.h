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
	std::vector<std::string> ParticleFiles;
	std::vector<std::string> InitialConditionFiles;
	std::string OutputFolder;

	double start_time = 0.0;
	double end_time = 0.0;
	double timestep = 0.0;
	double save_interval = 0.0;

	unsigned int number_of_CPU_threads;

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

	// Ultimate forces
	double ultimate_forces[3] = { 0.0, 0.0, 0.0 };
};

typedef std::vector<MaterialProperty> MaterialContainer;


//    ____                   _     _          _            _   _               _        
//   |  _ \    __ _   _ __  | |_  (_)   ___  | |   ___    | \ | |   ___     __| |   ___ 
//   | |_) |  / _` | | '__| | __| | |  / __| | |  / _ \   |  \| |  / _ \   / _` |  / _ \
//   |  __/  | (_| | | |    | |_  | | | (__  | | |  __/   | |\  | | (_) | | (_| | |  __/
//   |_|      \__,_| |_|     \__| |_|  \___| |_|  \___|   |_| \_|  \___/   \__,_|  \___|
//                                                                                      


class ParticleNode
{
public:
	// ID of the node
	int id;

	// Mass of the node
	double mass = 0.0;

	// Translational force on the node
	double3 force = make_double3(0.0, 0.0, 0.0);

	// Translational acceleration of the node
	double3 acceleration = make_double3(0.0, 0.0, 0.0);

	// Translational velocity of the node
	double3 velocity = make_double3(0.0, 0.0, 0.0);

	// Position of the node
	double3 position = make_double3(0.0, 0.0, 0.0);
};

typedef std::vector<ParticleNode> ParticleNodeContainer;

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

//    ____                   _     _          _            _____                       
//   |  _ \    __ _   _ __  | |_  (_)   ___  | |   ___    |  ___|   __ _    ___    ___ 
//   | |_) |  / _` | | '__| | __| | |  / __| | |  / _ \   | |_     / _` |  / __|  / _ \
//   |  __/  | (_| | | |    | |_  | | | (__  | | |  __/   |  _|   | (_| | | (__  |  __/
//   |_|      \__,_| |_|     \__| |_|  \___| |_|  \___|   |_|      \__,_|  \___|  \___|
//                                                                                     



class ParticleFace
{
public:
	// ID of the face
	int id;

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

	// Bonded
	int isBonded = 0;
};

typedef std::vector<ParticleFace> ParticleFaceContainer;

//    ____                   _     _          _        
//   |  _ \    __ _   _ __  | |_  (_)   ___  | |   ___ 
//   | |_) |  / _` | | '__| | __| | |  / __| | |  / _ \
//   |  __/  | (_| | | |    | |_  | | | (__  | | |  __/
//   |_|      \__,_| |_|     \__| |_|  \___| |_|  \___|
//                                                     


class Particle
{
public:
	// ID of the particle
	int id;

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

	// Bond indices
	int bonds[4] = { 0, 0, 0, 0 };

	// Number of active bonds
	int number_active_bonds = 0;

	// Center point of the particle
	double3 barycenter = make_double3(0.0, 0.0, 0.0);

	// Circumsphere radius
	double circumsphere_radius = 0.0;

	// Particle volume
	double volume = 0.0;

	// Particle total mass
	double mass = 0.0;
};

typedef std::vector<Particle> ParticleContainer;


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

	// Dependent node indices
	int nodes[2] = { 0 , 0 };

	// Spring stiffness
	double stiffness = 0.0;

	// Spring damping
	double damping = 0.0;

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



//    ____                        _              _      ____                   _                    _   
//   | __ )    ___    _ __     __| |   ___    __| |    / ___|   ___    _ __   | |_    __ _    ___  | |_ 
//   |  _ \   / _ \  | '_ \   / _` |  / _ \  / _` |   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|
//   | |_) | | (_) | | | | | | (_| | |  __/ | (_| |   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_ 
//   |____/   \___/  |_| |_|  \__,_|  \___|  \__,_|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|
//                                                                                                      


class BondedContact
{
public:
	// ID of the current bond
	int id;

	// Bond status
	int isActive = 0;

	// Dependent surface indices
	int faces[2] = { 0, 0 };

	// Dependent node indices
	int face_a_nodes[3] = { 0, 0, 0 };
	int face_b_nodes[3] = { 0, 0, 0 };

	// Status of the node-node contacts
	int isNodePairActive[3] = { 0, 0, 0 };

	// Normal length
	double normal_length[3] = { 0.0, 0.0, 0.0 };

	// Tangential 1 length
	double tangential_1_length[3] = { 0.0, 0.0, 0.0 };

	// Tangential 2 length
	double tangential_2_length[3] = { 0.0, 0.0, 0.0 };

	// Normal stiffness
	double normal_stiffness = 0.0;

	// Normal damping
	double normal_damping = 0.0;

	// Tangential 1 stiffness
	double tangential_1_stiffness = 0.0;

	// Tangential 1 damping
	double tangential_1_damping = 0.0;

	// Tangential 2 stiffness
	double tangential_2_stiffness = 0.0;

	// Tangential 2 damping
	double tangential_2_damping = 0.0;

	// Spring forces
	double spring_forces[3] = { 0.0, 0.0, 0.0 };

	// Damping forces
	double damping_forces[3] = { 0.0, 0.0, 0.0 };

	// Total forces
	double total_forces[3] = { 0.0, 0.0, 0.0 };

	// Ultimate normal force (magnitude)
	double ultimate_normal_force = 0.0;

	// Ultimate shear force (magnitude)
	double ultimate_shear_force = 0.0;

	// Breakage under normal load
	int normal_breakage[3] = { 0, 0, 0 };

	// Breakage under shear load
	int shear_breakage[3] = { 0, 0, 0 };
};

typedef std::vector<BondedContact> BondedContactContainer;