#pragma once

#include <vector>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <thread>

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
static int _particle_id = 0;
static int _axial_spring_id = 0;
static int _rotational_spring_id = 0;
static int _external_force_id = 0;


//    ____    _                       _           _     _                     ____                                         _     _              
//   / ___|  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __     |  _ \   _ __    ___    _ __     ___   _ __  | |_  (_)   ___   ___ 
//   \___ \  | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \    | |_) | | '__|  / _ \  | '_ \   / _ \ | '__| | __| | |  / _ \ / __|
//    ___) | | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |   |  __/  | |    | (_) | | |_) | |  __/ | |    | |_  | | |  __/ \__ \
//   |____/  |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|   |_|     |_|     \___/  | .__/   \___| |_|     \__| |_|  \___| |___/
//                                                                                                  |_|                                         



static unsigned int _simulation_step = 0;
static unsigned int _total_simulation_steps = 0;
static unsigned int _save_interval = 0;

static double _simulation_time = 0.0;

//    ___      __   ___     ___                       _   _                 
//   |_ _|    / /  / _ \   / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __  ___ 
//    | |    / /  | | | | | | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \/ __|
//    | |   / /   | |_| | | |_| | |_) |  __/ | | (_| | |_| | (_) | | | \__ \
//   |___| /_/     \___/   \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|___/
//                              |_|                                         



__host__ void loadSettings(Settings& settings);


__host__ void loadMaterialProperties(Settings&			settings, 
									 MaterialContainer& materials);

__host__ void loadNodesParticles(Settings&				settings, 
								 ParticleNodeContainer& nodes, 
								 ParticleContainer&		particles);




__host__ void writeParticleNodes(ParticleNodeContainer	nodes,
								 Settings				settings,
								 unsigned int			step);

__host__ void writeParticleFaces(ParticleFaceContainer	faces,
								 ParticleNodeContainer	nodes,
								 Settings				settings,
								 unsigned int			step);

__host__ void writeParticles(ParticleContainer		particles,
							 ParticleNodeContainer	nodes, 
							 Settings				settings, 
							 unsigned int			step);

__host__ void wrtieParticleAxialSprings(AxialSpringContainer		axial_springs,
										ParticleContainer			particles,
										IntersectionNodeContainer	intersection_nodes,
										Settings					settings,
										unsigned int				step);

__host__ void writeParticleRotationalSprings(ParticleContainer			particles,
											 RotationalSpringContainer	rotational_springs,
											 AxialSpringContainer		axial_springs,
											 IntersectionNodeContainer	intersection_nodes,
											 Settings					settings,
											 unsigned int				step);


//    ____                                 _          ___                                  _     _                       
//   / ___|    ___    __ _   _ __    ___  | |__      / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   \___ \   / _ \  / _` | | '__|  / __| | '_ \    | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    ___) | |  __/ | (_| | | |    | (__  | | | |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |____/   \___|  \__,_| |_|     \___| |_| |_|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                          |_|                                                          


__host__ __device__ int findClosestParticleNodeToLocation(ParticleNode* nodes,
														  unsigned int	number_of_nodes,
														  double3		location);

__host__ __device__ int findParticleNodeInLocation(ParticleNode*	nodes,
												   unsigned int		number_of_nodes,
												   double3			location);

int findParticleNode(ParticleNodeContainer& nodes, int node_id);
int findParticleFace(ParticleFaceContainer& faces, int face_id);
int findMaterialProperty(MaterialContainer& materials, int material_id);
int findParticleIntersectionNode(IntersectionNodeContainer& intersection_nodes, int node_id);


//    ____                   _     _          _            _   _               _             ___                                  _     _                       
//   |  _ \    __ _   _ __  | |_  (_)   ___  | |   ___    | \ | |   ___     __| |   ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |_) |  / _` | | '__| | __| | |  / __| | |  / _ \   |  \| |  / _ \   / _` |  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   |  __/  | (_| | | |    | |_  | | | (__  | | |  __/   | |\  | | (_) | | (_| | |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_|      \__,_| |_|     \__| |_|  \___| |_|  \___|   |_| \_|  \___/   \__,_|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                 |_|                                                          

__host__ void calculateParticleNodeMass(Particle*		particles,
										ParticleNode*	nodes,
										unsigned int	particle_index);


__host__ __device__ int isParticleNodeOnPlane(ParticleNode* nodes,
											  unsigned int	node_index,
											  double3		location,
											  double3		normal);


__host__ __device__ void integrateParticleNode(ParticleNode*	nodes,
											   unsigned int		node_index,
										       const double		timestep);


__host__ void resetParticleNodesCPU(ParticleNode*	nodes,
									unsigned int	number_of_nodes,
									unsigned int	number_of_threads);


__host__ void updateParticleNodesCPU(ParticleNode*	nodes,
									 unsigned int	number_of_nodes,
									 unsigned int	number_of_threads,
									 double			timestep);




//    ___           _                                       _     _                     _   _               _             ___                                  _     _                       
//   |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                                              |_|                                                          


__host__ void createParticleIntersectionNodes(IntersectionNodeContainer&	intersection_nodes, 
											  ParticleNodeContainer&		nodes, 
											  ParticleFaceContainer&		faces, 
											  ParticleContainer&			particles,
											  MaterialContainer&			materials);

__host__ __device__ void calculateParticleIntersectionNodePosition(IntersectionNode*	intersection_nodes, 
																   ParticleNode*		nodes,
																   unsigned int			inode_index);


__host__ __device__ void calculateParticleIntersectionNodeVelocity(IntersectionNode*	intersection_nodes,
																   ParticleNode*		nodes,
																   unsigned int			inode_index);



__host__ void updateParticleIntersectionNodesCPU(IntersectionNode*	intersection_nodes,
												 unsigned int		number_of_inodes,
												 ParticleNode*		nodes,
												 unsigned int		number_of_threads);

//    ____                   _     _          _            _____                            ___                                  _     _                       
//   |  _ \    __ _   _ __  | |_  (_)   ___  | |   ___    |  ___|   __ _    ___    ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |_) |  / _` | | '__| | __| | |  / __| | |  / _ \   | |_     / _` |  / __|  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   |  __/  | (_| | | |    | |_  | | | (__  | | |  __/   |  _|   | (_| | | (__  |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_|      \__,_| |_|     \__| |_|  \___| |_|  \___|   |_|      \__,_|  \___|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                |_|                                                          


__host__ void createParticleFaces(ParticleFaceContainer&	faces, 
								  ParticleContainer&		particles, 
								  ParticleNodeContainer&	nodes);



__host__ __device__ void calculateParticleFaceCenter(ParticleFace*	faces, 
													 ParticleNode*	nodes,
													 unsigned int	face_index);


__host__ __device__ void calculateParticleFaceNormal(ParticleFace* faces,
													 ParticleNode* nodes,
													 unsigned int  face_index);


__host__ __device__ void calculateParticleFaceArea(ParticleFace* faces,
												   ParticleNode* nodes,
												   unsigned int  face_index);



__host__ __device__ bool pointOnParticleFace(ParticleFace*	faces,
											 ParticleNode*	nodes, 
											 double*		coefficients, 
											 unsigned int	face_index,
											 double3		point);

__host__ void updateParticleFacesCPU(ParticleFace*	faces,
									 unsigned int	number_of_faces,
									 ParticleNode*	nodes,
									 unsigned int	number_of_threads);


//    ____                   _     _          _             ___                                  _     _                       
//   |  _ \    __ _   _ __  | |_  (_)   ___  | |   ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |_) |  / _` | | '__| | __| | |  / __| | |  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   |  __/  | (_| | | |    | |_  | | | (__  | | |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_|      \__,_| |_|     \__| |_|  \___| |_|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                |_|                                                          

__host__ __device__ void calculateParticleCenter(Particle*		particles,
												 ParticleNode*	nodes, 
												 unsigned int	particle_index);



__host__ void assignParticleMaterial(Particle*					particles,
									 unsigned int				particle_index,
									 MaterialProperty*			materials,
									 unsigned int				size_of_materials);



__host__ __device__ void calculateParticleVolume(Particle*		particles,
												 ParticleNode*	nodes,
												 unsigned int	particle_index);



__host__ void calculateParticleMass(Particle*			particles,
									MaterialProperty*	materials,
									unsigned int		particle_index);




__host__ __device__ void calculateParticleCircumsphere(Particle*		particles,
													   ParticleNode*	nodes,
													   unsigned int		particle_index);


__host__ __device__ void checkParticleDamage(Particle* particles,
	unsigned int				number_of_particles,
	ParticleFace* faces,
	IntersectionNode* intersection_nodes,
	AxialSpring* axial_springs,
	RotationalSpring* rotational_springs);



__host__ void initializeParticles(Particle*					particles,
								  unsigned int				number_of_particles,
								  ParticleNode*				nodes,
								  MaterialProperty*			materials,
								  unsigned int				size_of_materials,
								  Settings&					settings);

__host__ void updateParticlesCPU(Particle*		particles,
								 unsigned int	number_of_particles,
								 ParticleNode*	nodes,
								 unsigned int	number_of_threads);




//       _             _           _     ____                   _                      ___                                  _     _                       
//      / \    __  __ (_)   __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//     / _ \   \ \/ / | |  / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    / ___ \   >  <  | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   /_/   \_\ /_/\_\ |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                              |_|                         |___/            |_|                                                          

__host__ void createAxialSprings(AxialSpringContainer&		axial_springs, 
								 ParticleContainer&			particles, 
								 IntersectionNodeContainer& intersection_nodes, 
								 MaterialContainer&			materials);

__host__ __device__ void calculateAxialSpring(AxialSpring*		axial_springs, 
											  IntersectionNode* intersection_nodes,
											  unsigned int		spring_index);

__host__ __device__ void applyAxialSpringForces(AxialSpring*		axial_springs,
												unsigned int		number_of_springs,
												IntersectionNode*	intersection_nodes,
												ParticleNode*		nodes);

__host__ void updateAxialSpringsCPU(AxialSpring*		axial_springs,
									unsigned int		number_of_springs,
									IntersectionNode*	intersection_nodes,
									unsigned int		number_of_threads);

//    ____            _             _     _                           _     ____                   _                      ___                                  _     _                       
//   |  _ \    ___   | |_    __ _  | |_  (_)   ___    _ __     __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   | |_) |  / _ \  | __|  / _` | | __| | |  / _ \  | '_ \   / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` |   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   |  _ <  | (_) | | |_  | (_| | | |_  | | | (_) | | | | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_| \_\  \___/   \__|  \__,_|  \__| |_|  \___/  |_| |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, |    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                 |_|                         |___/            |_|                                                          

__host__ void createRotationalSprings(RotationalSpringContainer&	rotational_springs,
									  ParticleContainer&			particles,
									  AxialSpringContainer&			axial_springs,
									  IntersectionNodeContainer&	intersection_nodes,
									  MaterialContainer&			materials);

__host__ __device__ void calculateRotationalSpring(RotationalSpring*	rotational_springs,
												   AxialSpring*			axial_springs,
												   IntersectionNode*	intersection_nodes,
												   unsigned int			rspring_index);

__host__ __device__ void applyRotationalSpringForces(RotationalSpring*	rotational_springs,
													 unsigned int		number_of_rsprings,
													 AxialSpring*		axial_springs,
													 IntersectionNode*  intersection_nodes,
													 ParticleNode*		nodes);

__host__ void updateRotationalSpringsCPU(RotationalSpring*	rotational_springs,
										 unsigned int		number_of_rsprings,
										 AxialSpring*		axial_springs,
										 IntersectionNode*	intersection_nodes,
										 unsigned int		number_of_threads);


//    ___           _   _     _           _      ____                       _   _   _     _                       
//   |_ _|  _ __   (_) | |_  (_)   __ _  | |    / ___|   ___    _ __     __| | (_) | |_  (_)   ___    _ __    ___ 
//    | |  | '_ \  | | | __| | |  / _` | | |   | |      / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \  / __|
//    | |  | | | | | | | |_  | | | (_| | | |   | |___  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | | \__ \
//   |___| |_| |_| |_|  \__| |_|  \__,_| |_|    \____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                                


__host__ void applyInitialConditions(ParticleNodeContainer& nodes,
									 Settings&				settings);



//    ____                                _                             ____                       _   _   _     _                       
//   | __ )    ___    _   _   _ __     __| |   __ _   _ __   _   _     / ___|   ___    _ __     __| | (_) | |_  (_)   ___    _ __    ___ 
//   |  _ \   / _ \  | | | | | '_ \   / _` |  / _` | | '__| | | | |   | |      / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \  / __|
//   | |_) | | (_) | | |_| | | | | | | (_| | | (_| | | |    | |_| |   | |___  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | | \__ \
//   |____/   \___/   \__,_| |_| |_|  \__,_|  \__,_| |_|     \__, |    \____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_| |___/
//                                                           |___/                                                                       


__host__ void applyBoundaryConditions(ParticleNodeContainer& nodes,
									  Settings& settings);





//    _____          _                                   _     _____                                   ___                                  _     _                       
//   | ____| __  __ | |_    ___   _ __   _ __     __ _  | |   |  ___|   ___    _ __    ___    ___     / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   |  _|   \ \/ / | __|  / _ \ | '__| | '_ \   / _` | | |   | |_     / _ \  | '__|  / __|  / _ \   | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//   | |___   >  <  | |_  |  __/ | |    | | | | | (_| | | |   |  _|   | (_) | | |    | (__  |  __/   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |_____| /_/\_\  \__|  \___| |_|    |_| |_|  \__,_| |_|   |_|      \___/  |_|     \___|  \___|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                                           |_|                                                          


__host__ void createExternalForces(ExternalForceContainer& external_forces,
								   ParticleNodeContainer& nodes,
								   Settings& settings);

__host__ __device__ void applyExternalForces(ExternalForce*	external_forces,
											 unsigned int	number_of_forces,
											 ParticleNode*	nodes,
											 double			time);


//    ____    _                       _           _     _                      ___                                  _     _                       
//   / ___|  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __      / _ \   _ __     ___   _ __    __ _  | |_  (_)   ___    _ __    ___ 
//   \___ \  | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \    | | | | | '_ \   / _ \ | '__|  / _` | | __| | |  / _ \  | '_ \  / __|
//    ___) | | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |   | |_| | | |_) | |  __/ | |    | (_| | | |_  | | | (_) | | | | | \__ \
//   |____/  |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|    \___/  | .__/   \___| |_|     \__,_|  \__| |_|  \___/  |_| |_| |___/
//                                                                                   |_|                                                          

__host__ void processInputFiles(Settings&				settings,
								MaterialContainer&		materials, 
								ParticleNodeContainer&	nodes, 
								ParticleContainer&		particles);

__host__ void initializeSimulation(ParticleContainer& particles,
	ParticleFaceContainer& faces,
	ParticleNodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	MaterialContainer& materials,
	Settings& settings);

__host__ void runSimulation(ParticleContainer& particles,
	ParticleFaceContainer& faces,
	ParticleNodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	Settings& settings);

__host__ void exportSimulationData(ParticleContainer particles,
								   ParticleFaceContainer faces,
								   ParticleNodeContainer nodes,
								   IntersectionNodeContainer intersection_nodes,
								   AxialSpringContainer axial_springs,
								   RotationalSpringContainer rotational_springs,
								   Settings settings,
								   unsigned int step);




int main();

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

