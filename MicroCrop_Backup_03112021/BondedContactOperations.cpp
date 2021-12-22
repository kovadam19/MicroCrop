#include "Simulation.h"

__host__ void createBondedContacts(Particle*			particles,
								   unsigned int			number_of_particles,
								   ParticleFace*		faces,
								   ParticleNode*		nodes,
								   MaterialProperty*	materials,
								   BondedContact*		bonds)
{
	// Go through the particles
	for (int i = 0; i < number_of_particles - 1; i++)
	{
		// Check if the number of active bonds is less than four -> the particle is not fully bonded
		if (particles[i].number_active_bonds < 4)
		{
			// Compare the current particle to all the others
			for (int j = i; j < number_of_particles; j++)
			{
				// Check if the number of active bonds is less than four -> the particle is not fully bonded
				if (particles[j].number_active_bonds < 4)
				{
					// At this point we know none of the particles of interest is fully bonded
					// Let's check the free surfaces of the particles against each other
					for (int k = 0; k < 4; k++)
					{
						// Get the face index
						int face_a_index = particles[i].faces[k];

						// Check if the current face is NOT bonded
						if (faces[face_a_index].isBonded == 0)
						{
							// Check this face against the others of the other particle
							for (int m = 0; m < 4; m++)
							{
								// Get the face index
								int face_b_index = particles[j].faces[m];

								// Check if this faces is NOT bonded
								if (faces[face_b_index].isBonded == 0)
								{
									// Let's find the adjacent nodes
									int number_of_adjacent_nodes = 0;
									int face_a_nodes[3] = { -1, -1, -1 };
									int face_b_nodes[3] = { -1, -1, -1 };

									findAdjacentFaceNodes(faces,
														   nodes,
														   face_a_index,
														   face_b_index,
														   &number_of_adjacent_nodes,
														   face_a_nodes,
														   face_b_nodes);

									// Check if there are three adjacent nodes
									if (number_of_adjacent_nodes == 3)
									{
										// Create a new bonded contact
										BondedContact new_bond;

										// Set the ID for the contact
										new_bond.id = _bonded_contact_id++;

										// Set it to be active
										new_bond.isActive = 1;

										// Assign the faces
										new_bond.faces[0] = face_a_index;
										new_bond.faces[1] = face_b_index;

										// Set the faces to be bonded
										faces[face_a_index].isBonded = 1;
										faces[face_b_index].isBonded = 1;

										// Assign the nodes to the bond
										new_bond.face_a_nodes[0] = face_a_nodes[0];
										new_bond.face_a_nodes[1] = face_a_nodes[1];
										new_bond.face_a_nodes[2] = face_a_nodes[2];
										new_bond.face_b_nodes[0] = face_b_nodes[0];
										new_bond.face_b_nodes[1] = face_b_nodes[1];
										new_bond.face_b_nodes[2] = face_b_nodes[2];

										// Set the node pairs to be active
										new_bond.isNodePairActive[0] = 1;
										new_bond.isNodePairActive[1] = 1;
										new_bond.isNodePairActive[2] = 1;

										// Get the material indices
										int particle_a_mat_index = particles[i].material_property;
										int particle_b_mat_index = particles[j].material_property;

										// Calculate the material stiffness vectors
										double3 particle_a_stiffness_0 = materials[particle_a_mat_index].axes[0] * materials[particle_a_mat_index].axial_stiffnesses[0];
										double3 particle_a_stiffness_1 = materials[particle_a_mat_index].axes[1] * materials[particle_a_mat_index].axial_stiffnesses[1];
										double3 particle_a_stiffness_2 = materials[particle_a_mat_index].axes[2] * materials[particle_a_mat_index].axial_stiffnesses[2];
										double3 particle_b_stiffness_0 = materials[particle_b_mat_index].axes[0] * materials[particle_b_mat_index].axial_stiffnesses[0];
										double3 particle_b_stiffness_1 = materials[particle_b_mat_index].axes[1] * materials[particle_b_mat_index].axial_stiffnesses[1];
										double3 particle_b_stiffness_2 = materials[particle_b_mat_index].axes[2] * materials[particle_b_mat_index].axial_stiffnesses[2];

										// Get the normal of the bond
										double3 normal = faces[face_a_index].normal;

										// Calcualte the first tangential direction based on the barycenter and node A of face A
										double3 face_a_barycenter = faces[face_a_index].barycenter;
										double3 face_a_node_a_position = nodes[faces[face_a_index].nodes[0]].position;
										double3 tangential_1 = get_normalize(face_a_node_a_position - face_a_barycenter);

										// Calculate the second tangential direction based on the normal and first tangential direction
										double3 tangential_2 = get_normalize(cross(normal, tangential_1));
										
										// Calculate the projection of the stiffness vectors to the bond normal vector
										



									}
								}
							}
						}
					}
				}
			}
		}
	}
}