#include "Simulation.h"

__host__ void createIntersectionNodes(IntersectionNodeContainer&	intersection_nodes,
											  NodeContainer&				nodes,
											  FaceContainer&				faces,
											  CellContainer&				cells,
											  MaterialContainer&			materials)
{
	// Go through all the cells
	for (int i = 0; i < cells.size(); i++)
	{
		// Get the cell material index
		int material_index = cells[i].material_property;

		// Number of intersection points per cell
		int nodes_per_cell = 0;

		// Go through the anisotropy axes
		for (int j = 0; j < 3; j++)
		{
			// Get the current axis
			double3 axis = materials[material_index].axes[j];

			// Number of intersection nodes
			int nodes_per_axis = 0;

			// Go through the faces
			for (int k = 0; k < 4; k++)
			{
				// Get the face index
				int face_index = cells[i].faces[k];

				// Get the position of one of the nodes on the face
				double3 node_a_position = nodes[faces[face_index].nodes[0]].position;

				// Calculate a vector from the cell barycenter to one of the nodes on the face
				double3 center_to_node = node_a_position - cells[i].barycenter;

				// Get the face normal
				double3 face_normal = faces[face_index].normal;

				// Calculate the normal component to the face normal
				double vector_normal_component = dot(center_to_node, face_normal);

				// Calculate the normal component of the axis to the face normal
				double axis_normal_component = dot(axis, face_normal);

				// Calculate the distance from the barycenter to the intersection point
				double distance = vector_normal_component / axis_normal_component;

				// Calculate the intersection point position
				double3 intersection_point = cells[i].barycenter + axis * distance;

				// Check if the point falls within the triangle
				double coefficients[3] = { 0.0, 0.0, 0.0 };

				bool point_on_face = pointOnFace(&faces[0],
														 &nodes[0],
														 coefficients,
														 cells[i].faces[k],
														 intersection_point);

				if (point_on_face)
				{
					// Check if this is the first intersection node
					//if (nodes_per_axis == 0)
					//{
						// We simply create the intersection node
						// Create a new intersection node
						IntersectionNode new_node;

						// Assign an ID to the intersection node
						new_node.id = _intersection_node_id++;

						// Set the status to be active
						new_node.status = 1;

						// Assign the cell to the node
						new_node.cell = i;

						// Assign the nodes to the intersection node
						new_node.nodes[0] = faces[face_index].nodes[0];
						new_node.nodes[1] = faces[face_index].nodes[1];
						new_node.nodes[2] = faces[face_index].nodes[2];

						// Assign the coefficients for the shape function
						new_node.coefficients[0] = coefficients[0];
						new_node.coefficients[1] = coefficients[1];
						new_node.coefficients[2] = coefficients[2];

						// Get the independent node positions
						double3 node_a_position = nodes[new_node.nodes[0]].position;
						double3 node_b_position = nodes[new_node.nodes[1]].position;
						double3 node_c_position = nodes[new_node.nodes[2]].position;

						// Calculate the position of the the intersection node
						new_node.position = new_node.coefficients[0] * node_a_position +
											new_node.coefficients[1] * node_b_position +
											new_node.coefficients[2] * node_c_position;

						// Add the new node to the intersection node container
						intersection_nodes.push_back(new_node);

						// Add the intersection node to the cell
						cells[i].intersections[nodes_per_cell] = intersection_nodes.size() - 1;

						// Increase the number of intersection nodes
						nodes_per_axis++;
						nodes_per_cell++;
					//}
					//else
					//{
					//	// We already have an intersection node
					//	// We have to check if the new one is not in the same location 
					//	// This can happen because of rounding error if the intersection point falls right on an edge or node

					//	// So get the last intersection node position
					//	int last_index = intersection_nodes.size() - 1;
					//	double3 last_node_position = intersection_nodes[last_index].position;

					//	// Get the facial node positions 
					//	double3 node_a_position = nodes[faces[face_index].nodes[0]].position;
					//	double3 node_b_position = nodes[faces[face_index].nodes[1]].position;
					//	double3 node_c_position = nodes[faces[face_index].nodes[2]].position;

					//	// Calculate the new node position
					//	double3 new_node_position = coefficients[0] * node_a_position +
					//								coefficients[1] * node_b_position +
					//								coefficients[2] * node_c_position;

					//	// Calculate the distance between the last and new intersection nodes
					//	double distance = length(last_node_position - new_node_position);

					//	// Check if the distance is NOT zero
					//	if (!isZero(distance))
					//	{
					//		// Create a new intersection node
					//		IntersectionNode new_node;

					//		// Assign an ID to the intersection node
					//		new_node.id = _intersection_node_id++;

					//		// Set the status to be active
					//		new_node.status = 1;

					//		// Assign the cell to the node
					//		new_node.cell = i;

					//		// Assign the nodes to the intersection node
					//		new_node.nodes[0] = faces[face_index].nodes[0];
					//		new_node.nodes[1] = faces[face_index].nodes[1];
					//		new_node.nodes[2] = faces[face_index].nodes[2];

					//		// Assign the coefficients for the shape function
					//		new_node.coefficients[0] = coefficients[0];
					//		new_node.coefficients[1] = coefficients[1];
					//		new_node.coefficients[2] = coefficients[2];

					//		// Get the independent node positions
					//		double3 node_a_position = nodes[new_node.nodes[0]].position;
					//		double3 node_b_position = nodes[new_node.nodes[1]].position;
					//		double3 node_c_position = nodes[new_node.nodes[2]].position;

					//		// Calculate the position of the the intersection node
					//		new_node.position = new_node.coefficients[0] * node_a_position +
					//							new_node.coefficients[1] * node_b_position +
					//							new_node.coefficients[2] * node_c_position;

					//		// Add the new node to the intersection node container
					//		intersection_nodes.push_back(new_node);

					//		// Add the intersection node to the cell
					//		cells[i].intersections[nodes_per_cell] = intersection_nodes.size() - 1;

					//		// Increase the number of intersection nodes
					//		nodes_per_axis++;
					//		nodes_per_cell++;
					//	}
					//}
				}

				// Check if we found all the nodes for the current axis
				if (nodes_per_axis == 2)
				{
					// Break the loop
					break;
				}
			}
		}
	}
}


__host__ __device__ void calculateIntersectionNodePosition(IntersectionNode*	intersection_nodes,
																   Node*		nodes,
																   int			inode_index)
{
	// Get the independent node positions
	double3 node_a_position = nodes[intersection_nodes[inode_index].nodes[0]].position;
	double3 node_b_position = nodes[intersection_nodes[inode_index].nodes[1]].position;
	double3 node_c_position = nodes[intersection_nodes[inode_index].nodes[2]].position;

	// Calculate the position of the the intersection node
	intersection_nodes[inode_index].position = intersection_nodes[inode_index].coefficients[0] * node_a_position +
											   intersection_nodes[inode_index].coefficients[1] * node_b_position +
											   intersection_nodes[inode_index].coefficients[2] * node_c_position;
}


__host__ __device__ void calculateIntersectionNodeVelocity(IntersectionNode*	intersection_nodes,
																   Node*		nodes,
																   int			inode_index)
{
	// Get the independent node velocities
	double3 node_a_velocity = nodes[intersection_nodes[inode_index].nodes[0]].velocity;
	double3 node_b_velocity = nodes[intersection_nodes[inode_index].nodes[1]].velocity;
	double3 node_c_velocity = nodes[intersection_nodes[inode_index].nodes[2]].velocity;

	// Calculate the velocity of the the intersection node
	intersection_nodes[inode_index].velocity = intersection_nodes[inode_index].coefficients[0] * node_a_velocity +
											   intersection_nodes[inode_index].coefficients[1] * node_b_velocity +
											   intersection_nodes[inode_index].coefficients[2] * node_c_velocity;
}



__host__ void updateIntersectionNodesCPU(IntersectionNode*	intersection_nodes,
												 int		number_of_inodes,
												 Node*		nodes,
												 int		number_of_threads)
{
	std::vector<std::thread> threads;

	auto update = [](IntersectionNode*	intersection_nodes,
					 int		number_of_inodes,
					 Node*		nodes,
					 const int			number_of_threads,
					 const int			thread_id)
	{
		int inode_index = thread_id;
		while (inode_index < number_of_inodes)
		{
			if (intersection_nodes[inode_index].status == 1)
			{
				calculateIntersectionNodePosition(intersection_nodes, nodes, inode_index);
				calculateIntersectionNodeVelocity(intersection_nodes, nodes, inode_index);
			}

			inode_index += number_of_threads;
		}
	};

	for (int thread_id = 0; thread_id < number_of_threads; thread_id++)
	{
		threads.push_back(std::thread(update,
									  intersection_nodes,
									  number_of_inodes,
									  nodes,
									  number_of_threads,
									  thread_id));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}
}