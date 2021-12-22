#pragma once

#include <vector>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

#include "Primitives.h"
#include "cudaVectorMath.h"


//    ____           _               _   _     _                    ___   ____        
//   |  _ \   _ __  (_)  _ __ ___   (_) | |_  (_) __   __   ___    |_ _| |  _ \   ___ 
//   | |_) | | '__| | | | '_ ` _ \  | | | __| | | \ \ / /  / _ \    | |  | | | | / __|
//   |  __/  | |    | | | | | | | | | | | |_  | |  \ V /  |  __/    | |  | |_| | \__ \
//   |_|     |_|    |_| |_| |_| |_| |_|  \__| |_|   \_/    \___|   |___| |____/  |___/
//                                                                                    

static int _material_id = 0;
static int _material_type_id = 0;
static int _node_id = 0;
static int _intersection_node_id = 0;
static int _face_id = 0;
static int _cell_id = 0;
static int _axial_spring_id = 0;
static int _rotational_spring_id = 0;
static int _external_force_id = 0;
static int _contact_id = 0;






//    ___      __   ___     ___                       _   _                 
//   |_ _|    / /  / _ \   / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __  ___ 
//    | |    / /  | | | | | | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \/ __|
//    | |   / /   | |_| | | |_| | |_) |  __/ | | (_| | |_| | (_) | | | \__ \
//   |___| /_/     \___/   \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|___/
//                              |_|                                         



__host__ void loadSettings(Settings& settings);


__host__ void loadMaterialProperties(Settings&			settings, 
									 MaterialContainer& materials);

__host__ void loadCells(Settings&		settings, 
						NodeContainer&	nodes, 
						CellContainer&	cells);




__host__ void writeCellNodes(NodeContainer	nodes,
							 Settings		settings,
							 int			step,
							 std::string	file_name);

__host__ void writeCellFaces(FaceContainer	faces,
							 NodeContainer	nodes,
							 Settings		settings,
							 int			step,
							 std::string	file_name);

__host__ void writeCells(CellContainer&		cells,
						 NodeContainer&		nodes, 
						 Settings&			settings, 
						 int				step,
						 std::string		file_name);

__host__ void writeAxialSprings(AxialSpringContainer		axial_springs,
								IntersectionNodeContainer	intersection_nodes,
								Settings					settings,
								int							step,
								std::string					file_name);

__host__ void writeRotationalSprings(RotationalSpringContainer	rotational_springs,
									 AxialSpringContainer		axial_springs,
									 IntersectionNodeContainer	intersection_nodes,
									 Settings					settings,
									 int						step,
									 std::string				file_name);


//    ____                                 _          ___                                  _     _                       
//   / ___|    ___    __ _   _ __    ___  | |__      / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   \___ \   / _ \  / _` | | '__|  / __| | '_ \    | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    ___) | |  __/ | (_| | | |    | (__  | | | |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |____/   \___|  \__,_| |_|     \___| |_| |_|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                          |_|                                                          


__host__ __device__ int findClosestNodeToLocation(Node*			nodes,
												  int	number_of_nodes,
												  double3		location);

__host__ __device__ int findNodeInLocation(Node*	nodes,
										   int		number_of_nodes,
										   double3			location);

int findParticleNode(NodeContainer& nodes, int node_id);
int findParticleFace(FaceContainer& faces, int face_id);
int findMaterialProperty(MaterialContainer& materials, int material_id);
int findParticleIntersectionNode(IntersectionNodeContainer& intersection_nodes, int node_id);



//    _   _               _             ___                                  _     _                       
//   | \ | |   ___     __| |   ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   |  \| |  / _ \   / _` |  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   | |\  | | (_) | | (_| | |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_| \_|  \___/   \__,_|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                            |_|                                                          


__host__ void calculateNodeMass(Cell*			cells,
								Node*			nodes,
								int				cell_index);


__host__ __device__ int isNodeOnPlane(Node*			nodes,
									  int			node_index,
									  double3		location,
									  double3		normal);


__host__ __device__ void integrateNode(Node*			nodes,
									   int				node_index,
									   const double		timestep);


__host__ void resetNodesCPU(Node*	        nodes,
							int				number_of_nodes,
							int				number_of_threads);


__host__ void updateNodesCPU(Node*		nodes,
							 int		number_of_nodes,
							 int		number_of_threads,
							 double		timestep);

__global__ void resetNodesCUDA(Node*	nodes,
							   int		number_of_nodes);


//    ___           _                                       _     _                     _   _               _             ___                                  _     _                       
//   |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                                              |_|                                                          


__host__ void createIntersectionNodes(IntersectionNodeContainer&	intersection_nodes, 
									  NodeContainer&				nodes, 
									  FaceContainer&				faces, 
									  CellContainer&				cells,
									  MaterialContainer&			materials);

__host__ __device__ void calculateIntersectionNodePosition(IntersectionNode*	intersection_nodes, 
														   Node*				nodes,
														   int			inode_index);


__host__ __device__ void calculateIntersectionNodeVelocity(IntersectionNode*	intersection_nodes,
														   Node*				nodes,
														   int			inode_index);



__host__ void updateIntersectionNodesCPU(IntersectionNode*	intersection_nodes,
										 int		number_of_inodes,
										 Node*				nodes,
										 int		number_of_threads);


//    _____                            ___                                  _     _                       
//   |  ___|   __ _    ___    ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |_     / _` |  / __|  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   |  _|   | (_| | | (__  |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_|      \__,_|  \___|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                           |_|                                                          


__host__ void createFaces(FaceContainer&	faces, 
						  CellContainer&	cells, 
						  NodeContainer&	nodes);



__host__ __device__ void calculateFaceCenter(Face*			faces, 
											 Node*			nodes,
											 int	face_index);


__host__ __device__ void calculateFaceNormal(Face*			faces,
											 Node*			nodes,
											 int	face_index);


__host__ __device__ void calculateFaceArea(Face*		 faces,
										   Node*		 nodes,
										   int  face_index);



__host__ __device__ bool pointOnFace(Face*			faces,
									 Node*			nodes, 
									 double*		coefficients, 
									 int	face_index,
									 double3		point);

__host__ void updateFacesCPU(Face*			faces,
							 int	number_of_faces,
							 Node*			nodes,
							 int	number_of_threads);



//     ____          _   _      ___                                  _     _                       
//    / ___|   ___  | | | |    / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |      / _ \ | | | |   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   | |___  |  __/ | | | |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//    \____|  \___| |_| |_|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                    |_|                                                          


__host__ __device__ void calculateCellCenter(Cell*			cells,
											 Node*			nodes, 
											 int	cell_index);



__host__ void assignCellMaterial(Cell*					cells,
								 int			cell_index,
								 MaterialProperty*		materials,
								 int			size_of_materials);



__host__ __device__ void calculateCellVolume(Cell*			cells,
											 Node*			nodes,
											 int	cell_index);



__host__ void calculateCellMass(Cell*				cells,
								MaterialProperty*	materials,
								int		cell_index);




__host__ __device__ void calculateCellCircumsphere(Cell*			cells,
												   Node*			nodes,
												   int		cell_index);


__host__ __device__ void checkCellDamage(Cell*				cells,
										 int		number_of_cells,
										 Face*				faces,
										 IntersectionNode*	intersection_nodes,
										 AxialSpring*		axial_springs,
										 RotationalSpring*	rotational_springs);



__host__ void initializeCells(Cell*					cells,
							  int			number_of_cells,
							  Node*					nodes,
							  MaterialProperty*		materials,
							  int			size_of_materials,
							  Settings&				settings);

__host__ void updateCellsCPU(Cell*			cells,
							 int	number_of_cells,
							 Node*			nodes,
							IntersectionNode* intersection_nodes,
							Face* faces,
							AxialSpring* axial_springs,
							RotationalSpring* rotational_springs,
							 int	number_of_threads);




//       _             _           _     ____                   _                      ___                                  _     _                       
//      / \    __  __ (_)   __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//     / _ \   \ \/ / | |  / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    / ___ \   >  <  | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   /_/   \_\ /_/\_\ |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                              |_|                         |___/            |_|                                                          

__host__ void createAxialSprings(AxialSpringContainer&		axial_springs, 
								 CellContainer&				cells, 
								 IntersectionNodeContainer& intersection_nodes, 
								 MaterialContainer&			materials);

inline __host__ __device__ void calculateAxialSpring(AxialSpring* axial_springs,
	IntersectionNode* intersection_nodes,
	int		spring_index)
{
	//    ___          _   _     _          _      ___          _               _          _     _                   
	//   |_ _|  _ _   (_) | |_  (_)  __ _  | |    / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//    | |  | ' \  | | |  _| | | / _` | | |   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//   |___| |_||_| |_|  \__| |_| \__,_| |_|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//                                                                                                               

	// Get the node indicies
	int node_a_index = axial_springs[spring_index].nodes[0];
	int node_b_index = axial_springs[spring_index].nodes[1];

	// Get the node positions
	double3 node_a_position = intersection_nodes[node_a_index].position;
	double3 node_b_position = intersection_nodes[node_b_index].position;

	// Claculate the distance vector from node A to node B
	double3 distance_vector = node_b_position - node_a_position;

	// Calculate the unit vector from node A to node B
	double3 unit_vector = get_normalize(distance_vector);

	// Calculte the current length
	double current_length = length(distance_vector);

	// Save the current length
	axial_springs[spring_index].current_length = current_length;

	// Get the initial length
	double initial_length = axial_springs[spring_index].initial_length;

	// Calculate the delta length
	double delta_length = current_length - initial_length;

	//    ___                _                   ___                             ___          _               _          _     _                   
	//   / __|  _ __   _ _  (_)  _ _    __ _    | __|  ___   _ _   __   ___     / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//   \__ \ | '_ \ | '_| | | | ' \  / _` |   | _|  / _ \ | '_| / _| / -_)   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//   |___/ | .__/ |_|   |_| |_||_| \__, |   |_|   \___/ |_|   \__| \___|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//         |_|                     |___/                                                                                                       

	// Get the spring stiffness
	double stiffness = axial_springs[spring_index].stiffness;

	// Calculate the magnitude force
	double force_magnitude = stiffness * fabs(delta_length);

	// Spring force vector
	double3 spring_force_vector = unit_vector * force_magnitude;

	// Check if the delta length is smaller then zero
	if (delta_length < 0.0)
	{
		// Compression
		// Spring force on node_a (since the unit vector points to node_b and we are in compression we have to multiply by -1 the node force on node_a)
		axial_springs[spring_index].spring_force_node_a = -spring_force_vector;

		// Spring force on node_b
		axial_springs[spring_index].spring_force_node_b = spring_force_vector;
	}
	else if (delta_length > 0.0)
	{
		// Tension
		// Spring force on node_a
		axial_springs[spring_index].spring_force_node_a = spring_force_vector;

		// Spring force on node_b (since the unit vector points to node_b and we are in tension we have to multiply by -1 the node force on node_b)
		axial_springs[spring_index].spring_force_node_b = -spring_force_vector;
	}

	//    ___                          _                   ___                             ___          _               _          _     _                   
	//   |   \   __ _   _ __    _ __  (_)  _ _    __ _    | __|  ___   _ _   __   ___     / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//   | |) | / _` | | '  \  | '_ \ | | | ' \  / _` |   | _|  / _ \ | '_| / _| / -_)   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//   |___/  \__,_| |_|_|_| | .__/ |_| |_||_| \__, |   |_|   \___/ |_|   \__| \___|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//                         |_|               |___/                                                                                                       

	// Get the damping
	double damping = axial_springs[spring_index].damping;

	// Get the node velocities
	double3 node_a_velocity = intersection_nodes[node_a_index].velocity;
	double3 node_b_velocity = intersection_nodes[node_b_index].velocity;

	// Calculate the relative velocity of node_a and node_b
	double3 relative_velocity = node_b_velocity - node_a_velocity;

	// Calculate the normal relative velocity of node_a and node_b
	double3 normal_relative_velocity = unit_vector * dot(unit_vector, relative_velocity);

	// Calculate the normal damping force magnitude
	double damping_force_magnitude = damping * length(normal_relative_velocity);

	// Check the direction of the relative normal velocity
	if (dot(normal_relative_velocity, unit_vector) < 0.0)
	{
		// In this case the nodes are getting closer to each other (loading)
		axial_springs[spring_index].damping_force_node_a = -1 * unit_vector * damping_force_magnitude;
		axial_springs[spring_index].damping_force_node_b = unit_vector * damping_force_magnitude;
	}
	else
	{
		// In this case the nodes are getting further from each other (unloading)
		axial_springs[spring_index].damping_force_node_a = unit_vector * damping_force_magnitude;
		axial_springs[spring_index].damping_force_node_b = -1 * unit_vector * damping_force_magnitude;
	}

	//    _____         _            _     ___                             ___          _               _          _     _                   
	//   |_   _|  ___  | |_   __ _  | |   | __|  ___   _ _   __   ___     / __|  __ _  | |  __   _  _  | |  __ _  | |_  (_)  ___   _ _    ___
	//     | |   / _ \ |  _| / _` | | |   | _|  / _ \ | '_| / _| / -_)   | (__  / _` | | | / _| | || | | | / _` | |  _| | | / _ \ | ' \  (_-<
	//     |_|   \___/  \__| \__,_| |_|   |_|   \___/ |_|   \__| \___|    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_| /__/
	//                                                                                                                                       

	// Calculte the total force on node_a and node_b
	axial_springs[spring_index].total_force_node_a = axial_springs[spring_index].spring_force_node_a + axial_springs[spring_index].damping_force_node_a;
	axial_springs[spring_index].total_force_node_b = axial_springs[spring_index].spring_force_node_b + axial_springs[spring_index].damping_force_node_b;
}

__host__ __device__ void applyAxialSpringForces(AxialSpring*		axial_springs,
												int		number_of_springs,
												IntersectionNode*	intersection_nodes,
												Node*				nodes);

__host__ void updateAxialSpringsCPU(AxialSpring*		axial_springs,
									int		number_of_springs,
									IntersectionNode*	intersection_nodes,
									int		number_of_threads);

__global__ void updateAxialSpringsCUDA(AxialSpring* axial_springs,
										int		number_of_springs,
										IntersectionNode* intersection_nodes);

//    ____            _             _     _                           _     ____                   _                      ___                                  _     _                       
//   |  _ \    ___   | |_    __ _  | |_  (_)   ___    _ __     __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |_) |  / _ \  | __|  / _` | | __| | |  / _ \  | '_ \   / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   |  _ <  | (_) | | |_  | (_| | | |_  | | | (_) | | | | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_| \_\  \___/   \__|  \__,_|  \__| |_|  \___/  |_| |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                 |_|                         |___/            |_|                                                          

__host__ void createRotationalSprings(RotationalSpringContainer&	rotational_springs,
									  CellContainer&				cells,
									  AxialSpringContainer&			axial_springs,
									  IntersectionNodeContainer&	intersection_nodes,
									  MaterialContainer&			materials);

__host__ __device__ void calculateRotationalSpring(RotationalSpring*	rotational_springs,
												   AxialSpring*			axial_springs,
												   IntersectionNode*	intersection_nodes,
												   int			rspring_index);

__host__ __device__ void applyRotationalSpringForces(RotationalSpring*	rotational_springs,
													 int		number_of_rsprings,
													 AxialSpring*		axial_springs,
													 IntersectionNode*  intersection_nodes,
													 Node*		nodes);

__host__ void updateRotationalSpringsCPU(RotationalSpring*	rotational_springs,
										 int		number_of_rsprings,
										 AxialSpring*		axial_springs,
										 IntersectionNode*	intersection_nodes,
										 int		number_of_threads);


//    ___           _   _     _           _      ____                       _   _   _     _                       
//   |_ _|  _ __   (_) | |_  (_)   __ _  | |    / ___|   ___    _ __     __| | (_) | |_  (_)   ___    _ __    ___ 
//    | |  | '_ \  | | | __| | |  / _` | | |   | |      / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \  / __|
//    | |  | | | | | | | |_  | | | (_| | | |   | |___  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | | \__ \
//   |___| |_| |_| |_|  \__| |_|  \__,_| |_|    \____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                                


__host__ void applyInitialConditions(NodeContainer&		nodes,
									 Settings&			settings);



//    ____                                _                             ____                       _   _   _     _                       
//   | __ )    ___    _   _   _ __     __| |   __ _   _ __   _   _     / ___|   ___    _ __     __| | (_) | |_  (_)   ___    _ __    ___ 
//   |  _ \   / _ \  | | | | | '_ \   / _` |  / _` | | '__| | | | |   | |      / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \  / __|
//   | |_) | | (_) | | |_| | | | | | | (_| | | (_| | | |    | |_| |   | |___  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | | \__ \
//   |____/   \___/   \__,_| |_| |_|  \__,_|  \__,_| |_|     \__, |    \____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_| |___/
//                                                           |___/                                                                       


__host__ void applyBoundaryConditions(NodeContainer&	nodes,
									  Settings&			settings);





//    _____          _                                   _     _____                                   ___                                  _     _                       
//   | ____| __  __ | |_    ___   _ __   _ __     __ _  | |   |  ___|   ___    _ __    ___    ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   |  _|   \ \/ / | __|  / _ \ | '__| | '_ \   / _` | | |   | |_     / _ \  | '__|  / __|  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   | |___   >  <  | |_  |  __/ | |    | | | | | (_| | | |   |  _|   | (_) | | |    | (__  |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_____| /_/\_\  \__|  \___| |_|    |_| |_|  \__,_| |_|   |_|      \___/  |_|     \___|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                           |_|                                                          


__host__ void createExternalForces(ExternalForceContainer&	external_forces,
								   NodeContainer&			nodes,
								   Settings&				settings);

__host__ __device__ void applyExternalForces(ExternalForce*	external_forces,
											 int	number_of_forces,
											 Node*			nodes,
											 double			time);




//     ____                   _                    _        ___                                  _     _                       
//    / ___|   ___    _ __   | |_    __ _    ___  | |_     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_    | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                |_|                                                          

__host__ void initializeContacts(ContactContainer& contacts,
								 CellContainer& particles);

__host__ __device__ void detectContacts(Contact* contacts,
										Cell* particles,
										int number_of_particles,
										Face* faces,
										Node* nodes,
										MaterialProperty* materials);

//    ____    _                       _           _     _                      ___                                  _     _                       
//   / ___|  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __      / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   \___ \  | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \    | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    ___) | | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |____/  |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                   |_|                                                          

__host__ void processInputFiles(Settings&				settings,
								MaterialContainer&		materials, 
								NodeContainer&			nodes, 
								CellContainer&			cells);

__host__ void initializeSimulation(CellContainer& particles,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	ContactContainer& contacts,
	MaterialContainer& materials,
	Settings& settings);

__host__ void checkSimulation(CellContainer& cells,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	MaterialContainer& materials,
	Settings& settings);


__host__ void runSimulationCPU(CellContainer& cells,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	Settings& settings);


__host__ cudaError_t runSimulationCUDA(CellContainer& host_cells,
	FaceContainer& host_faces,
	NodeContainer& host_nodes,
	IntersectionNodeContainer& host_intersection_nodes,
	AxialSpringContainer& host_axial_springs,
	RotationalSpringContainer& host_rotational_springs,
	ExternalForceContainer& host_external_forces,
	Settings& host_settings);


__host__ void exportSimulationData(CellContainer& cells,
	FaceContainer& faces,
	NodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	Settings& settings,
	int step);


__global__ void addKernel(int* c, const int* a, const int* b);
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);


//    __  __           _         
//   |  \/  |   __ _  (_)  _ __  
//   | |\/| |  / _` | | | | '_ \ 
//   | |  | | | (_| | | | | | | |
//   |_|  |_|  \__,_| |_| |_| |_|
//                               



int main();