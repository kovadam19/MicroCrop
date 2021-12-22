#include "Simulation.h"


__host__ __device__ int findClosestParticleNodeToLocation(ParticleNode*	nodes,
												   unsigned int		number_of_nodes,
												   double3			location)
{
	// Create a variable with infinite distance
	double min_distance = INF;

	// Create a variable for the node index
	int node_index = -1;

	// Go through the nodes and find the closest one to the given location
	for (int j = 0; j < number_of_nodes; j++)
	{
		// Calculate the distance between the node and the given location
		double distance = length(location - nodes[j].position);

		// Check if the current node is closer to the location
		if (distance < min_distance)
		{
			// Update the minimum distance and the node index
			min_distance = distance;
			node_index = j;
		}
	}

	// Return the node index
	return node_index;
}


__host__ __device__ int findParticleNodeInLocation(ParticleNode* nodes,
												   unsigned int	 number_of_nodes,
												   double3		 location)
{
	// Create a variable for the node index
	int node_index = -1;

	// Go through the nodes and find the node that is exactly in the location
	for (int j = 0; j < number_of_nodes; j++)
	{
		// Check if the current node is in the given location
		if (nodes[j].position == location)
		{
			// Set the node index
			node_index = j;

			// Break the loop
			break;
		}
	}

	// Return the node index
	return node_index;
}

// Find a particle node among the nodes based on its ID
int findParticleNode(ParticleNodeContainer& nodes, int node_id)
{
	// Go through the nodes
	for (int i = 0; i < nodes.size(); i++)
	{
		// Check if the ID matches with the ID to find
		if (nodes[i].id == node_id) return i;
	}

	// If we reach this point then the node is not on the list
	return -1;
}

// Find a paricle face based on its ID
int findParticleFace(ParticleFaceContainer& faces, int face_id)
{
	// Go through the faces
	for (int i = 0; i < faces.size(); i++)
	{
		// Check if the ID matches with the ID to find
		if (faces[i].id == face_id) return i;
	}

	// If we reach this point then the face is not on the list
	return -1;
}

// Find a material property among the materials based on its ID
int findMaterialProperty(MaterialContainer& materials, int material_id)
{
	// Go through the materials
	for (int i = 0; i < materials.size(); i++)
	{
		// Check if the ID matches with the ID to find
		if (materials[i].id == material_id) return i;
	}

	// If we reach this point then the material is not on the list
	return -1;
}

// Find an intersectino node amond the nodes based on its ID
int findParticleIntersectionNode(IntersectionNodeContainer& intersection_nodes, int node_id)
{
	// Go through the nodes
	for (int i = 0; i < intersection_nodes.size(); i++)
	{
		// Check if the ID matches with the ID to find
		if (intersection_nodes[i].id == node_id) return i;
	}

	// If we reach this point then the node is not on the list
	return -1;
}