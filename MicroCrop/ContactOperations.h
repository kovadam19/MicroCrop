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
#include <thread>

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Other includes
#include "Primitives.h"
#include "CellOperations.h"
#include "SearchOperations.h"
#include "FaceOperations.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _   _     _           _   _                   ____                   _                    _         
//   |_ _|  _ __   (_) | |_  (_)   __ _  | | (_)  ____   ___     / ___|   ___    _ __   | |_    __ _    ___  | |_   ___ 
//    | |  | '_ \  | | | __| | |  / _` | | | | | |_  /  / _ \   | |      / _ \  | '_ \  | __|  / _` |  / __| | __| / __|
//    | |  | | | | | | | |_  | | | (_| | | | | |  / /  |  __/   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_  \__ \
//   |___| |_| |_| |_|  \__| |_|  \__,_| |_| |_| /___|  \___|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__| |___/
//                                                                                                                      
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Initializes a given number of contacts
inline __host__ void initializeContacts(ContactContainer&	contacts, 
										const int			number_of_contacts)
{
	// Initialize the container
	for (int i = 0; i < number_of_contacts; i++)
	{
		// Create an empty contact
		Contact new_contact;

		// Assign the ID
		new_contact.id = _contact_id++;

		// Add it to the container
		contacts.push_back(new_contact);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____           _                   _        ____                   _                    _   
//   |  _ \    ___  | |_    ___    ___  | |_     / ___|   ___    _ __   | |_    __ _    ___  | |_ 
//   | | | |  / _ \ | __|  / _ \  / __| | __|   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|
//   | |_| | |  __/ | |_  |  __/ | (__  | |_    | |___  | (_) | | | | | | |_  | (_| | | (__  | |_ 
//   |____/   \___|  \__|  \___|  \___|  \__|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|
//                                                                                                
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Detects contact between a given node and the cells
inline __host__ __device__ void detectContact(Contact*				contacts,
											  const int				contact_index,
											  Cell*					cells,
											  const int				number_of_cells,
											  Face*					faces,
											  Node*					nodes,
											  InteractionProperty*	interactions,
											  const int				number_of_interactions)
{
	// Check if the contact is passive
	if (contacts[contact_index].status == 0)
	{
		// Get the node position
		double3 node_position = nodes[contact_index].position;

		// Go through the cells
		for (int i = 0; i < number_of_cells; i++)
		{
			// Check if the cell is active
			if (cells[i].status == 1)
			{
				// Check if the node does NOT belong to the cell
				if (contact_index != cells[i].nodes[0] &&
					contact_index != cells[i].nodes[1] &&
					contact_index != cells[i].nodes[2] &&
					contact_index != cells[i].nodes[3])
				{
					// Check if the node is within the cell (there is a positive overlap)
					int closest_face_index = pointInCell(cells, i, faces, node_position);
					if (closest_face_index != -1)
					{
						// Set the contact status to be active
						contacts[contact_index].status = 1;

						// Set the face index
						contacts[contact_index].cell_face = closest_face_index;

						// Get the components
						int component_a = faces[closest_face_index].component;
						int component_b = nodes[contact_index].component;

						// Find the interaction property
						int interaction_index = findInteractionProperty(interactions,
							number_of_interactions,
							component_a,
							component_b);

						// Assign the interaction properties
						if (interaction_index != -1)
						{
							// Assign the coefficient of static friction to the contact
							contacts[contact_index].coefficient_of_static_friction = interactions[interaction_index].coefficient_of_static_priction;

							// Assign the normal stiffness to the contact
							contacts[contact_index].normal_stiffness = interactions[interaction_index].normal_stiffness;

							// Assign the tangential stiffness to the contact
							contacts[contact_index].tangential_stiffness = interactions[interaction_index].tangential_stiffness;
						}

						// Break the loop
						break;
					}
				}
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____           _                   _        ____                   _                    _        ____   ____    _   _ 
//   |  _ \    ___  | |_    ___    ___  | |_     / ___|   ___    _ __   | |_    __ _    ___  | |_     / ___| |  _ \  | | | |
//   | | | |  / _ \ | __|  / _ \  / __| | __|   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|   | |     | |_) | | | | |
//   | |_| | |  __/ | |_  |  __/ | (__  | |_    | |___  | (_) | | | | | | |_  | (_| | | (__  | |_    | |___  |  __/  | |_| |
//   |____/   \___|  \__|  \___|  \___|  \__|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|    \____| |_|      \___/ 
//                                                                                                                          
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Distributes contact detection among multiple CPU threads
inline __host__ void detectContactCPU(Contact*				contacts,
									  const int				number_of_contacts,
									  Cell*					cells,
									  const int				number_of_cells,
									  Face*					faces,
									  Node*					nodes,
									  InteractionProperty*	interactions,
									  const int				number_of_interactions,
									  const int				number_of_threads)
{
	// Creating a container for the threads
	std::vector<std::thread> threads;

	// Creating a lambda function for the contact detection
	auto detect = [](Contact*				contacts,
					 int					number_of_contacts,
					 Cell*					cells,
					 int					number_of_cells,
					 Face*					faces,
					 Node*					nodes,
					 InteractionProperty*	interactions,
					 int					number_of_interactions,
					 const int				number_of_threads,
					 const int				thread_id)
	{
		// Go through the contacts
		for (int contact_index = thread_id; contact_index < number_of_contacts; contact_index += number_of_threads)
		{
			detectContact(contacts, contact_index, cells, number_of_cells, faces, nodes, interactions, number_of_interactions);
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(detect,
									  contacts,
									  number_of_contacts,
									  cells,
									  number_of_cells,
									  faces,
									  nodes,
									  interactions,
									  number_of_interactions,
									  number_of_threads,
									  thread_id));
	}

	// Joining threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _               ____                   _                    _   
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___     / ___|   ___    _ __   | |_    __ _    ___  | |_ 
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_ 
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|
//                                                                                                                   
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates the contact forces for the given contact based on the applied physics
inline __host__ __device__ void calculateContact(Contact*		contacts,
												 const int		contact_index,
												 Face*			faces,
												 Node*			nodes,
												 const double	timestep)
{
	// Get the position of the node
	double3 node_position = nodes[contact_index].position;

	// Get the face index
	int face_index = contacts[contact_index].cell_face;

	// Get the face normal
	double3 face_normal = faces[face_index].normal;

	// Get the face distance
	double face_distance = faces[face_index].distance;

	// Calculate the normal overlap
	double normal_overlap = -1.0 * (dot(face_normal, node_position) - face_distance);

	// Check if the normal overlap is larger than zero
	if (normal_overlap > 0)
	{
		// Assign the normal overlap to the contact
		contacts[contact_index].normal_overlap = normal_overlap;

		// Get the node velocity
		double3 node_velocity = nodes[contact_index].velocity;

		// Get the face velocity in the barycenter
		double3 face_velocity = calculateFaceCenterVelocity(faces, nodes, face_index);

		// Calculate the relative velocity
		double3 relative_velocity = node_velocity - face_velocity;

		// Calculate the normal velocity
		double3 normal_velocity = face_normal * dot(face_normal, relative_velocity);

		// Calculate the tangential velocity
		double3 tangential_velocity = relative_velocity - normal_velocity;

		// Calculate the tangential overlap
		double3 tangential_overlap = contacts[contact_index].tangential_overlap + (tangential_velocity * timestep);

		// Assign the tangential overlap to the contact
		contacts[contact_index].tangential_overlap = tangential_overlap;

		// Calculate the normal spring force
		double3 normal_spring_force = face_normal * contacts[contact_index].normal_stiffness * normal_overlap;

		// Assign the normal force to the contact
		contacts[contact_index].normal_force = normal_spring_force;

		// Calculate the tangential force
		double3 tangential_spring_force = tangential_overlap * contacts[contact_index].tangential_stiffness * -1.0;

		// Check if we are in a sliding situation (tangential force exceeds the Coulomb firction force)
		if (length(tangential_spring_force) > (length(normal_spring_force) * contacts[contact_index].coefficient_of_static_friction))
		{
			// Calculate the tangential unit vector
			double3 tangential_unit_vector = get_normalize(tangential_spring_force);

			// Calculate the maximum sliding tangential force
			tangential_spring_force = tangential_unit_vector * length(normal_spring_force) * contacts[contact_index].coefficient_of_static_friction;

			// Calculate the tangential overlap
			double3 new_tangential_overlap = (tangential_spring_force / contacts[contact_index].tangential_stiffness) * -1.0;

			// Update the tangential overlap
			contacts[contact_index].tangential_overlap = new_tangential_overlap;
		}

		// Assign the tangential force to the contact
		contacts[contact_index].tangential_force = tangential_spring_force;

		// Calculate the total contact force
		contacts[contact_index].total_force = normal_spring_force + tangential_spring_force;
	}
	else
	{
		// Deactivate the contact
		contacts[contact_index].status = 0;
		contacts[contact_index].cell_face = 0;
		contacts[contact_index].coefficient_of_static_friction = 0.0;
		contacts[contact_index].normal_stiffness = 0.0;
		contacts[contact_index].tangential_stiffness = 0.0;
		contacts[contact_index].normal_overlap = 0.0;
		contacts[contact_index].tangential_overlap = make_double3(0.0, 0.0, 0.0);
		contacts[contact_index].normal_force = make_double3(0.0, 0.0, 0.0);
		contacts[contact_index].tangential_force = make_double3(0.0, 0.0, 0.0);
		contacts[contact_index].total_force = make_double3(0.0, 0.0, 0.0);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _               ____                   _                    _              ____   ____    _   _ 
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___     / ___|   ___    _ __   | |_    __ _    ___  | |_   ___     / ___| |  _ \  | | | |
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   | |      / _ \  | '_ \  | __|  / _` |  / __| | __| / __|   | |     | |_) | | | | |
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_  \__ \   | |___  |  __/  | |_| |
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__| |___/    \____| |_|      \___/ 
//                                                                                                                                                   
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Distributes contact calculations among multiple CPU threads
inline __host__ void calculateContactsCPU(Contact*		contacts,
										  const int		number_of_contacts,
										  Face*			faces,
										  Node*			nodes,
										  const double	timestep,
										  const int		number_of_threads)
{
	// Creating a container for the threads
	std::vector<std::thread> threads;

	// Creating a lambda function
	auto calculate = [](Contact*	contacts,
						int			number_of_contacts,
						Face*		faces,
						Node*		nodes,
						double		timestep,
						const int	number_of_threads,
						const int	thread_id)
	{
		// Go through the contacts
		for (int contact_index = thread_id; contact_index < number_of_contacts; contact_index += number_of_threads)
		{
			// Check if the contact is active
			if (contacts[contact_index].status == 1)
			{
				calculateContact(contacts, contact_index, faces, nodes, timestep);
			}
		}
	};

	// Creating threads
	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(calculate,
									  contacts,
									  number_of_contacts,
									  faces,
									  nodes,
									  timestep,
									  number_of_threads,
									  thread_id));
	}

	// Joining the threads
	for (auto& thread : threads)
	{
		thread.join();
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _                      _              ____                   _                    _       _____                                    
//      / \     _ __    _ __   | |  _   _     / ___|   ___    _ __   | |_    __ _    ___  | |_    |  ___|   ___    _ __    ___    ___   ___ 
//     / _ \   | '_ \  | '_ \  | | | | | |   | |      / _ \  | '_ \  | __|  / _` |  / __| | __|   | |_     / _ \  | '__|  / __|  / _ \ / __|
//    / ___ \  | |_) | | |_) | | | | |_| |   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_    |  _|   | (_) | | |    | (__  |  __/ \__ \
//   /_/   \_\ | .__/  | .__/  |_|  \__, |    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__|   |_|      \___/  |_|     \___|  \___| |___/
//             |_|     |_|          |___/                                                                                                   
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Applies the contact forces onto the nodes
inline __host__ void applyContactForces(Contact*	contacts,
										const int	number_of_contacts,
										Face*		faces,
										Node*		nodes)
{
	// Go through the contacts
	for (int i = 0; i < number_of_contacts; i++)
	{
		// Check if the contact is active
		if (contacts[i].status == 1)
		{
			// Apply the contact force onto the node
			nodes[i].force += contacts[i].total_force;

			// Apply the contact force onto the face nodes
			for (int j = 0; j < 3; j++)
			{
				nodes[faces[contacts[i].cell_face].nodes[j]].force += (-1 * contacts[i].total_force) / 3.0;
			}
		}
	}
}
