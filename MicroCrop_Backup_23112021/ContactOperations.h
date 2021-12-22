//#include "Simulation.h"
//
//__host__ void initializeContacts(ContactContainer& contacts, 
//								 CellContainer& particles)
//{
//	// Get the number of particles
//	int number_of_particles = particles.size();
//
//	// Create an empty contact
//	Contact new_contact;
//
//	// Reserve the place for passive contacts
//	contacts.reserve(number_of_particles);
//
//	// Initialize the container
//	for (int i = 0; i < number_of_particles * 3; i++)
//	{
//		contacts.push_back(new_contact);
//	}
//}
//
//__host__ __device__ void detectContacts(Contact* contacts,
//	Cell* particles,
//	int number_of_particles,
//	Face* faces,
//	Node* nodes,
//	MaterialProperty* materials)
//{
//	// Go through the particles
//	for (int i = 0; i < number_of_particles - 1; i++)
//	{
//		// And check against the rest
//		for (int j = i + 1; j < number_of_particles; j++)
//		{
//			// Get the position of the barycenter of cell A
//			double3 particle_a_position = particles[i].barycenter;
//
//			// Get the radius of the circumsphere of cell A
//			double particle_a_radius = particles[i].circumsphere_radius;
//
//			// Get the position of the barycenter of cell B
//			double3 particle_b_position = particles[j].barycenter;
//
//			// Get the radius of the circumsphere of cell B
//			double particle_b_radius = particles[j].circumsphere_radius;
//
//			// Caluclate the distance between the particles
//			double distance = length(particle_a_position - particle_b_position);
//
//			// Check if the distance is smaller than the two circumsphere radii
//			if (distance < (particle_a_radius + particle_b_radius))
//			{
//
//			}
//		}
//	}
//}