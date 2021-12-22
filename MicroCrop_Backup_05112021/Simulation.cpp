#include "Simulation.h"


__host__ void processInputFiles(Settings&				settings,
								MaterialContainer&		materials,
								ParticleNodeContainer&	nodes,
								ParticleContainer&		particles)
{
	loadSettings(settings);
	loadMaterialProperties(settings, materials);
	loadNodesParticles(settings, nodes, particles);
}

__host__ void initializeSimulation(ParticleContainer& particles,
									ParticleFaceContainer& faces,
								   ParticleNodeContainer& nodes,
									IntersectionNodeContainer& intersection_nodes,
									AxialSpringContainer& axial_springs,
									RotationalSpringContainer& rotational_springs,
									ExternalForceContainer& external_forces,
									MaterialContainer& materials,
									Settings& settings)
{
	initializeParticles(&particles[0],
						particles.size(),
						&nodes[0],
						&materials[0],
						materials.size(),
						settings);

	createParticleFaces(faces,
						particles,
						nodes);

	updateParticleFacesCPU(&faces[0],
						   faces.size(),
						   &nodes[0],
						   settings.number_of_CPU_threads);

	createParticleIntersectionNodes(intersection_nodes,
									nodes,
									faces, 
									particles, 
									materials);

	updateParticleIntersectionNodesCPU(&intersection_nodes[0],
									   intersection_nodes.size(),
									   &nodes[0],
									   settings.number_of_CPU_threads);

	createAxialSprings(axial_springs,
					   particles, 
					   intersection_nodes, 
					   materials);
	
	createRotationalSprings(rotational_springs, 
							particles, 
							axial_springs, 
							intersection_nodes, 
							materials);

	applyInitialConditions(nodes,
						   settings);

	applyBoundaryConditions(nodes,
						    settings);

	createExternalForces(external_forces,
						 nodes,
						 settings);


	_total_simulation_steps = int((settings.end_time - settings.start_time) / settings.timestep);
	_save_interval = int(settings.save_interval / settings.timestep);
}


__host__ void runSimulation(ParticleContainer& particles,
	ParticleFaceContainer& faces,
	ParticleNodeContainer& nodes,
	IntersectionNodeContainer& intersection_nodes,
	AxialSpringContainer& axial_springs,
	RotationalSpringContainer& rotational_springs,
	ExternalForceContainer& external_forces,
	Settings& settings)
{
	while (_simulation_step <= _total_simulation_steps)
	{

		if (std::fmod(_simulation_step, _save_interval) == 0)
		{
			std::cout << "Saving at time " << _simulation_time << " second." << std::endl;
			exportSimulationData(particles,
								 faces,
								 nodes,
								 intersection_nodes,
								 axial_springs,
								 rotational_springs,
								 settings,
								 _simulation_step);
		}

		checkParticleDamage(&particles[0], 
							particles.size(), 
							&faces[0], 
							&intersection_nodes[0], 
							&axial_springs[0], 
							&rotational_springs[0]);

		resetParticleNodesCPU(&nodes[0],
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
							_simulation_time);

		updateParticleNodesCPU(&nodes[0], 
							   nodes.size(), 
							   settings.number_of_CPU_threads, 
							   settings.timestep);

		updateParticleIntersectionNodesCPU(&intersection_nodes[0], 
										   intersection_nodes.size(),
										   &nodes[0], 
										   settings.number_of_CPU_threads);

		updateParticleFacesCPU(&faces[0], 
							   faces.size(), 
							   &nodes[0], 
							   settings.number_of_CPU_threads);

		updateParticlesCPU(&particles[0], 
						   particles.size(), 
						   &nodes[0], 
						   settings.number_of_CPU_threads);

		_simulation_step++;
		_simulation_time += settings.timestep;
	}

	exportSimulationData(particles,
						 faces,
						 nodes,
						 intersection_nodes,
						 axial_springs,
						 rotational_springs,
						 settings,
						 _simulation_step);
}



__host__ void exportSimulationData(ParticleContainer particles,
	ParticleFaceContainer faces,
	ParticleNodeContainer nodes,
	IntersectionNodeContainer intersection_nodes,
	AxialSpringContainer axial_springs,
	RotationalSpringContainer rotational_springs,
	Settings settings,
	unsigned int step)
{
	std::vector<std::thread> threads;


	threads.push_back(std::thread(writeParticleNodes, nodes, settings, step));
	threads.push_back(std::thread(writeParticleFaces, faces, nodes, settings, step));
	threads.push_back(std::thread(writeParticles, particles, nodes, settings, step));
	threads.push_back(std::thread(wrtieParticleAxialSprings, axial_springs, particles, intersection_nodes, settings, step));
	threads.push_back(std::thread(writeParticleRotationalSprings, particles, rotational_springs, axial_springs, intersection_nodes, settings, step));


	for (auto& thread : threads)
	{
		thread.join();
	}
}