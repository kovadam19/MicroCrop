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

#include "cudaVectorMath.h"
#include "IO_Operations.h"
#include "SearchOperations.h"
#include "Primitives.h"
#include "NodeOperations.h"
#include "IntersectionNodeOperations.h"
#include "FaceOperations.h"
#include "CellOperations.h"
#include "AxialSpringOperations.h"
#include "RotationalSpringOperations.h"
#include "InitialConditionOperations.h"
#include "BoundaryConditionOperations.h"
#include "ExternalForceOperations.h"


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
