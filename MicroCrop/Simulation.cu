/****************************************************************
* Project: MicroCrop - Advanced Anisotropic Mass-Spring System
* Author : Adam Kovacs
* Version : 1.0.0
* Maintainer : Adam Kovacs
* Email: kovadam19@gmail.com
* Released: 01 January 2022
*****************************************************************/

// Other includes
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
#include "ContactOperations.h"
#include "FixedVelocityOperations.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____            _                          _   
//   |  _ \    __ _  | |_    __ _   ___    ___  | |_ 
//   | | | |  / _` | | __|  / _` | / __|  / _ \ | __|
//   | |_| | | (_| | | |_  | (_| | \__ \ |  __/ | |_ 
//   |____/   \__,_|  \__|  \__,_| |___/  \___|  \__|
//                                                   
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Creating host data
Settings settings;
MaterialContainer materials;
InteractionPropertyContainer interaction_properties;
NodeContainer nodes;
IntersectionNodeContainer intersection_nodes;
FaceContainer faces;
CellContainer cells;
AxialSpringContainer axial_springs;
RotationalSpringContainer rotational_springs;
ExternalForceContainer external_forces;
ContactContainer contacts;

// Creating device data
Cell* dev_cells;
Face* dev_faces;
Node* dev_nodes;
IntersectionNode* dev_intersection_nodes;
AxialSpring* dev_axial_springs;
RotationalSpring* dev_rotational_springs;
Contact* dev_contacts;
InteractionProperty* dev_interaction_properties;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ___           _   _     _           _   _                  ____    _                       _           _     _                 
//   |_ _|  _ __   (_) | |_  (_)   __ _  | | (_)  ____   ___    / ___|  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __  
//    | |  | '_ \  | | | __| | |  / _` | | | | | |_  /  / _ \   \___ \  | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \ 
//    | |  | | | | | | | |_  | | | (_| | | | | |  / /  |  __/    ___) | | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |
//   |___| |_| |_| |_|  \__| |_|  \__,_| |_| |_| /___|  \___|   |____/  |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|
//                                                                                                                                   
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__host__ void initializeSimulation(CellContainer&               cells,
                                   FaceContainer&               faces,
                                   NodeContainer&               nodes,
                                   IntersectionNodeContainer&   intersection_nodes,
                                   AxialSpringContainer&        axial_springs,
                                   RotationalSpringContainer&   rotational_springs,
                                   ExternalForceContainer&      external_forces,
                                   ContactContainer&            contacts,
                                   MaterialContainer&           materials,
                                   Settings&                    settings)
{
    logData("Initializing cells...", settings);
    initializeCells(&cells[0],
        cells.size(),
        &nodes[0],
        &materials[0],
        materials.size(),
        settings);

    logData("Creating cell faces...", settings);
    createFaces(faces,
        cells,
        nodes);

    updateFacesCPU(&faces[0],
        faces.size(),
        &nodes[0],
        settings.number_of_CPU_threads);

    logData("Creating intersection nodes...", settings);
    createIntersectionNodes(intersection_nodes,
        nodes,
        faces,
        cells,
        materials);

    updateIntersectionNodesCPU(&intersection_nodes[0],
        intersection_nodes.size(),
        &nodes[0],
        settings.number_of_CPU_threads);

    logData("Creating axial springs...", settings);
    createAxialSprings(axial_springs,
        cells,
        intersection_nodes,
        materials);

    logData("Creating rotational springs...", settings);
    createRotationalSprings(rotational_springs,
        cells,
        axial_springs,
        intersection_nodes,
        materials);

    logData("Applying initial conditions...", settings);
    applyInitialConditions(nodes,
        settings);

    logData("Applying boundary conditions...", settings);
    applyBoundaryConditions(nodes,
        settings);

    logData("Applying fixed velocities...", settings);
    applyFixedVelocities(nodes,
        settings);

    logData("Creating external forces...", settings);
    createExternalForces(external_forces,
        nodes,
        settings);

    logData("Initializing the contacts...", settings);
    initializeContacts(contacts,
        nodes.size());
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____   _                     _        ____    _                       _           _     _                 
//    / ___| | |__     ___    ___  | | __   / ___|  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __  
//   | |     | '_ \   / _ \  / __| | |/ /   \___ \  | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \ 
//   | |___  | | | | |  __/ | (__  |   <     ___) | | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |
//    \____| |_| |_|  \___|  \___| |_|\_\   |____/  |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|
//                                                                                                               
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__host__ int checkSimulation(CellContainer&                cells,
                             FaceContainer&                faces,
                             NodeContainer&                nodes,
                             IntersectionNodeContainer&    intersection_nodes,
                             AxialSpringContainer&         axial_springs,
                             RotationalSpringContainer&    rotational_springs,
                             ExternalForceContainer&       external_forces,
                             MaterialContainer&            materials,
                             Settings&                     settings)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //     ___   _                 _     _                   ___         _     _     _                    
    //    / __| | |_    ___   __  | |__ (_)  _ _    __ _    / __|  ___  | |_  | |_  (_)  _ _    __ _   ___
    //   | (__  | ' \  / -_) / _| | / / | | | ' \  / _` |   \__ \ / -_) |  _| |  _| | | | ' \  / _` | (_-<
    //    \___| |_||_| \___| \__| |_\_\ |_| |_||_| \__, |   |___/ \___|  \__|  \__| |_| |_||_| \__, | /__/
    //                                             |___/                                       |___/      
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Check the number of components
    if (settings.number_of_components <= 0)
    {
        logData("ERROR: Number of components cannot be zero or negative", settings);
        return 1;
    }

    // Check the start and end time
    if (settings.end_time <= settings.start_time)
    {
        logData("ERROR: End time is smaller than the start time", settings);
        return 1;
    }

    // Check the time step
    if (settings.timestep <= 0.0)
    {
        logData("ERROR: Timestep cannot be zero or negative", settings);
        return 1;
    }

    // Check the number of CPU threads
    if (settings.number_of_CPU_threads <= 0)
    {
        logData("ERROR: Number of CPU threads cannot be zero or negative", settings);
        return 1;
    }

    // Check the number of GPU threads
    if (settings.GPU_threads_per_block <= 0)
    {
        logData("ERROR: Number of GPU threads cannot be zero or negative", settings);
        return 1;
    }

    // Check the number of GPU blocks
    if (settings.GPU_number_of_blocks <= 0)
    {
        logData("ERROR: Number of GPU blocks cannot be zero or negative", settings);
        return 1;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      _        _     _               _     _                                  _                 _          _      
    //     /_\    __| |   (_)  _  _   ___ | |_  (_)  _ _    __ _     _ __    __ _  | |_   ___   _ _  (_)  __ _  | |  ___
    //    / _ \  / _` |   | | | || | (_-< |  _| | | | ' \  / _` |   | '  \  / _` | |  _| / -_) | '_| | | / _` | | | (_-<
    //   /_/ \_\ \__,_|  _/ |  \_,_| /__/  \__| |_| |_||_| \__, |   |_|_|_| \__,_|  \__| \___| |_|   |_| \__,_| |_| /__/
    //                  |__/                               |___/                                                        
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Variable for the number of adjusted materials
    int number_adjusted_materials = 0;

    // Adjusted materials
    std::vector<int> adjusted_materials;

    // Finding corrupt rotational springs
    for (auto& rspring : rotational_springs)
    {
        // Check if the initial angle of the spring is not 90 deg
        if (rspring.initial_angle <= 1.57 || rspring.initial_angle >= 1.58)
        {
            // We have to rotate the anisotropy axes by a bit around each axis and recalculate the intersection points, axial and rotational springs
            // Get the cell index
            int cell_index = rspring.cell;

            // Get the material index
            int material_index = cells[cell_index].material_property;

            // Check if the material is already adjusted
            bool is_adjusted = false;
            for (auto& adjusted_material : adjusted_materials)
            {
                if (adjusted_material == material_index)
                {
                    is_adjusted = true;
                }
            }

            if (!is_adjusted)
            {
                // Add it to the adjusted materials
                adjusted_materials.push_back(material_index);

                // Increase the number of adjusted cells
                number_adjusted_materials++;

                // Rotation matrices
                double angle_x = settings.adjust_angle_x;
                double3 Rx_0 = make_double3(1.0, 0.0, 0.0);
                double3 Rx_1 = make_double3(0.0, cos(angle_x), -sin(angle_x));
                double3 Rx_2 = make_double3(0.0, sin(angle_x), cos(angle_x));

                double angle_y = settings.adjust_angle_y;
                double3 Ry_0 = make_double3(cos(angle_y), 0.0, sin(angle_y));
                double3 Ry_1 = make_double3(0.0, 1.0, 0.0);
                double3 Ry_2 = make_double3(-sin(angle_y), 0.0, cos(angle_y));

                double angle_z = settings.adjust_angle_z;
                double3 Rz_0 = make_double3(cos(angle_z), -sin(angle_z), 0.0);
                double3 Rz_1 = make_double3(sin(angle_z), cos(angle_z), 0.0);
                double3 Rz_2 = make_double3(0.0, 0.0, 1.0);

                // Go through the anisotropy axes
                for (int j = 0; j < 3; j++)
                {
                    // Get the current axis
                    double3 axis = materials[material_index].axes[j];

                    // Rotate the axis around X
                    axis.x = dot(axis, Rx_0);
                    axis.y = dot(axis, Rx_1);
                    axis.z = dot(axis, Rx_2);

                    // Rotate the axis around Y
                    axis.x = dot(axis, Ry_0);
                    axis.y = dot(axis, Ry_1);
                    axis.z = dot(axis, Ry_2);

                    // Rotate the axis around Z
                    axis.x = dot(axis, Rz_0);
                    axis.y = dot(axis, Rz_1);
                    axis.z = dot(axis, Rz_2);

                    // Set the new axis
                    materials[material_index].axes[j] = axis;
                }
            }
        }
    }

    // Check if the number of adjusted materials is larger than zero
    if (number_adjusted_materials > 0)
    {
        logData("WARNING: Corrupt materials found!", settings);
        logData("Deleting corrupt intersection nodes, axial and rotational springs...", settings);
        intersection_nodes.clear();
        axial_springs.clear();
        rotational_springs.clear();

        logData("Creating adjusted intersection nodes...", settings);
        createIntersectionNodes(intersection_nodes,
            nodes,
            faces,
            cells,
            materials);

        updateIntersectionNodesCPU(&intersection_nodes[0],
            intersection_nodes.size(),
            &nodes[0],
            settings.number_of_CPU_threads);

        logData("Creating adjusted axial springs...", settings);
        createAxialSprings(axial_springs,
            cells,
            intersection_nodes,
            materials);

        logData("Creating adjusted rotational springs...", settings);
        createRotationalSprings(rotational_springs,
            cells,
            axial_springs,
            intersection_nodes,
            materials);

        logData("Number of adjusted materials: " + std::to_string(number_adjusted_materials), settings);

        for (auto& adjusted_material : adjusted_materials)
        {
            logData("Adjusted material index: " + std::to_string(adjusted_material), settings);
            logData("Adjusted material component: " + std::to_string(materials[adjusted_material].component), settings);
            logData("Adjusted material location: " + std::to_string(materials[adjusted_material].location.x) + " " + std::to_string(materials[adjusted_material].location.y) + " " + std::to_string(materials[adjusted_material].location.z), settings);
        }
    }
    else
    {
        logData("No corrupt materials found", settings);
    }

    // If we reach this point then all is good
    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____           _       ____                   _               
//   / ___|    ___  | |_    |  _ \    ___  __   __ (_)   ___    ___ 
//   \___ \   / _ \ | __|   | | | |  / _ \ \ \ / / | |  / __|  / _ \
//    ___) | |  __/ | |_    | |_| | |  __/  \ V /  | | | (__  |  __/
//   |____/   \___|  \__|   |____/   \___|   \_/   |_|  \___|  \___|
//                                                                  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Sets the GPU device
__host__ cudaError_t setDevice(const int device_id)
{
    // Variable for CUDA status
    cudaError_t cudaStatus;

    // Set the CUDA device
    cudaStatus = cudaSetDevice(device_id);
    if (cudaStatus != cudaSuccess) logData("cudaSetDevice failed!", settings);

    return cudaStatus;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       _      _   _                          _              ____                   _                   __  __                                           
//      / \    | | | |   ___     ___    __ _  | |_    ___    |  _ \    ___  __   __ (_)   ___    ___    |  \/  |   ___   _ __ ___     ___    _ __   _   _ 
//     / _ \   | | | |  / _ \   / __|  / _` | | __|  / _ \   | | | |  / _ \ \ \ / / | |  / __|  / _ \   | |\/| |  / _ \ | '_ ` _ \   / _ \  | '__| | | | |
//    / ___ \  | | | | | (_) | | (__  | (_| | | |_  |  __/   | |_| | |  __/  \ V /  | | | (__  |  __/   | |  | | |  __/ | | | | | | | (_) | | |    | |_| |
//   /_/   \_\ |_| |_|  \___/   \___|  \__,_|  \__|  \___|   |____/   \___|   \_/   |_|  \___|  \___|   |_|  |_|  \___| |_| |_| |_|  \___/  |_|     \__, |
//                                                                                                                                                  |___/ 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Allocates memory on the GPU device
__host__ cudaError_t allocateDeviceMemory(void**    dev_cells,
                                          const int number_of_cells,
                                          void**    dev_faces, 
                                          const int number_of_faces,
                                          void**    dev_nodes,
                                          const int number_of_nodes,
                                          void**    dev_intersection_nodes,
                                          const int number_of_intersection_nodes,
                                          void**    dev_axial_springs,
                                          const int number_of_axial_springs,
                                          void**    dev_rotational_springs,
                                          const int number_of_rotational_springs,
                                          void**    dev_contacts,
                                          const int number_of_contacts,
                                          void**    dev_interaction_properties,
                                          const int number_of_interaction_properties)
{
    // Variable for CUDA status
    cudaError_t cudaStatus;
    
    // Allocate GPU buffers for device containers
    cudaStatus = cudaMalloc(dev_cells, number_of_cells * sizeof(Cell));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_cells!", settings);

    cudaStatus = cudaMalloc(dev_faces, number_of_faces * sizeof(Face));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_faces!", settings);

    cudaStatus = cudaMalloc(dev_nodes, number_of_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_nodes!", settings);

    cudaStatus = cudaMalloc(dev_intersection_nodes, number_of_intersection_nodes * sizeof(IntersectionNode));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_intersection_nodes!", settings);

    cudaStatus = cudaMalloc(dev_axial_springs, number_of_axial_springs * sizeof(AxialSpring));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_axial_springs!", settings);

    cudaStatus = cudaMalloc(dev_rotational_springs, number_of_rotational_springs * sizeof(RotationalSpring));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_rotational_springs!", settings);

    cudaStatus = cudaMalloc(dev_contacts, number_of_contacts * sizeof(Contact));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_contacts!", settings);

    cudaStatus = cudaMalloc(dev_interaction_properties, number_of_interaction_properties * sizeof(Contact));
    if (cudaStatus != cudaSuccess) logData("cudaMalloc failed on dev_interaction_properties!", settings);

    return cudaStatus;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____                             ____            _               _____             ____                   _               
//    / ___|   ___    _ __    _   _    |  _ \    __ _  | |_    __ _    |_   _|   ___     |  _ \    ___  __   __ (_)   ___    ___ 
//   | |      / _ \  | '_ \  | | | |   | | | |  / _` | | __|  / _` |     | |    / _ \    | | | |  / _ \ \ \ / / | |  / __|  / _ \
//   | |___  | (_) | | |_) | | |_| |   | |_| | | (_| | | |_  | (_| |     | |   | (_) |   | |_| | |  __/  \ V /  | | | (__  |  __/
//    \____|  \___/  | .__/   \__, |   |____/   \__,_|  \__|  \__,_|     |_|    \___/    |____/   \___|   \_/   |_|  \___|  \___|
//                   |_|      |___/                                                                                              
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Copies data to the GPU device
__host__ cudaError_t copyDataToDevice(void*         dev_cells, 
                                      const void*   host_cells,
                                      const int     number_of_cells,
                                      void*         dev_faces,
                                      const void*   host_faces,
                                      const int     number_of_faces,
                                      void*         dev_nodes,
                                      const void*   host_nodes,
                                      const int     number_of_nodes,
                                      void*         dev_intersection_nodes,
                                      const void*   host_intersection_nodes,
                                      const int     number_of_intersection_nodes,
                                      void*         dev_axial_springs,
                                      const void*   host_axial_springs,
                                      const int     number_of_axial_springs,
                                      void*         dev_rotational_springs,
                                      const void*   host_rotational_springs,
                                      const int     number_of_rotational_springs,
                                      void*         dev_contacts,
                                      const void*   host_contacts,
                                      const int     number_of_contacts,
                                      void*         dev_interaction_properties,
                                      const void*   host_interaction_properties,
                                      const int     number_of_interaction_properties)
{
    // Variable for CUDA status
    cudaError_t cudaStatus;

    // Copy containers from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_cells, host_cells, number_of_cells * sizeof(Cell), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_cells!", settings);

    cudaStatus = cudaMemcpy(dev_faces, host_faces, number_of_faces * sizeof(Face), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_faces!", settings);

    cudaStatus = cudaMemcpy(dev_nodes, host_nodes, number_of_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_nodes!", settings);

    cudaStatus = cudaMemcpy(dev_intersection_nodes, host_intersection_nodes, number_of_intersection_nodes * sizeof(IntersectionNode), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_intersection_nodes!", settings);

    cudaStatus = cudaMemcpy(dev_axial_springs, host_axial_springs, number_of_axial_springs * sizeof(AxialSpring), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_axial_springs!", settings);

    cudaStatus = cudaMemcpy(dev_rotational_springs, host_rotational_springs, number_of_rotational_springs * sizeof(RotationalSpring), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_rotational_springs!", settings);

    cudaStatus = cudaMemcpy(dev_contacts, host_contacts, number_of_contacts * sizeof(Contact), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_contacts!", settings);

    cudaStatus = cudaMemcpy(dev_interaction_properties, host_interaction_properties, number_of_interaction_properties * sizeof(InteractionProperty), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on dev_interaction_properties!", settings);

    return cudaStatus;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____                             ____            _               _____                                ____                   _               
//    / ___|   ___    _ __    _   _    |  _ \    __ _  | |_    __ _    |  ___|  _ __    ___    _ __ ___     |  _ \    ___  __   __ (_)   ___    ___ 
//   | |      / _ \  | '_ \  | | | |   | | | |  / _` | | __|  / _` |   | |_    | '__|  / _ \  | '_ ` _ \    | | | |  / _ \ \ \ / / | |  / __|  / _ \
//   | |___  | (_) | | |_) | | |_| |   | |_| | | (_| | | |_  | (_| |   |  _|   | |    | (_) | | | | | | |   | |_| | |  __/  \ V /  | | | (__  |  __/
//    \____|  \___/  | .__/   \__, |   |____/   \__,_|  \__|  \__,_|   |_|     |_|     \___/  |_| |_| |_|   |____/   \___|   \_/   |_|  \___|  \___|
//                   |_|      |___/                                                                                                                 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Copies data from the GPU device
__host__ cudaError_t copyDataFromDevice(const void* dev_cells,
                                        void*       host_cells,
                                        const int   number_of_cells,
                                        const void* dev_faces,
                                        void*       host_faces,
                                        const int   number_of_faces,
                                        const void* dev_nodes,
                                        void*       host_nodes,
                                        const int   number_of_nodes,
                                        const void* dev_intersection_nodes,
                                        void*       host_intersection_nodes,
                                        const int   number_of_intersection_nodes,
                                        const void* dev_axial_springs,
                                        void*       host_axial_springs,
                                        const int   number_of_axial_springs,
                                        const void* dev_rotational_springs,
                                        void*       host_rotational_springs,
                                        const int   number_of_rotational_springs,
                                        const void* dev_contacts,
                                        void*       host_contacts,
                                        const int   number_of_contacts,
                                        const void* dev_interaction_properties,
                                        void*       host_interaction_properties,
                                        const int   number_of_interaction_properties)
{
    // Variable for CUDA status
    cudaError_t cudaStatus;

    // Copy containers from GPU buffer to host memory
    cudaStatus = cudaMemcpy(host_nodes, dev_nodes, number_of_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_nodes!", settings);
    
    cudaStatus = cudaMemcpy(host_faces, dev_faces, number_of_faces * sizeof(Face), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_faces!", settings);
    
    cudaStatus = cudaMemcpy(host_cells, dev_cells, number_of_cells * sizeof(Cell), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_cells!", settings);

    cudaStatus = cudaMemcpy(host_intersection_nodes, dev_intersection_nodes, number_of_intersection_nodes * sizeof(IntersectionNode), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_intersection_nodes!", settings);
  
    cudaStatus = cudaMemcpy(host_axial_springs, dev_axial_springs, number_of_axial_springs * sizeof(AxialSpring), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_axial_springs!", settings);
    
    cudaStatus = cudaMemcpy(host_rotational_springs, dev_rotational_springs, number_of_rotational_springs * sizeof(RotationalSpring), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_rotational_springs!", settings);

    cudaStatus = cudaMemcpy(host_contacts, dev_contacts, number_of_contacts * sizeof(Contact), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_contacts!", settings);

    cudaStatus = cudaMemcpy(host_interaction_properties, dev_interaction_properties, number_of_interaction_properties * sizeof(InteractionProperty), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) logData("cudaMemcpy failed on host_interaction_properties!", settings);
    
    return cudaStatus;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____                        _       _   _               _                   ____   _   _   ____       _    
//   |  _ \    ___   ___    ___  | |_    | \ | |   ___     __| |   ___   ___     / ___| | | | | |  _ \     / \   
//   | |_) |  / _ \ / __|  / _ \ | __|   |  \| |  / _ \   / _` |  / _ \ / __|   | |     | | | | | | | |   / _ \  
//   |  _ <  |  __/ \__ \ |  __/ | |_    | |\  | | (_) | | (_| | |  __/ \__ \   | |___  | |_| | | |_| |  / ___ \ 
//   |_| \_\  \___| |___/  \___|  \__|   |_| \_|  \___/   \__,_|  \___| |___/    \____|  \___/  |____/  /_/   \_\
//                                                                                                               
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Resets the cell node on the GPU device
__global__ void resetNodesCUDA(Node*        nodes,
                               const int	number_of_nodes)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_nodes; i += stride)
    {
        nodes[i].force = make_double3(0.0, 0.0, 0.0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    ____           _                   _        ____                   _                    _              ____   _   _   ____       _    
//   |  _ \    ___  | |_    ___    ___  | |_     / ___|   ___    _ __   | |_    __ _    ___  | |_   ___     / ___| | | | | |  _ \     / \   
//   | | | |  / _ \ | __|  / _ \  / __| | __|   | |      / _ \  | '_ \  | __|  / _` |  / __| | __| / __|   | |     | | | | | | | |   / _ \  
//   | |_| | |  __/ | |_  |  __/ | (__  | |_    | |___  | (_) | | | | | | |_  | (_| | | (__  | |_  \__ \   | |___  | |_| | | |_| |  / ___ \ 
//   |____/   \___|  \__|  \___|  \___|  \__|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__| |___/    \____|  \___/  |____/  /_/   \_\
//                                                                                                                                          
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Detects contacts on the GPU device
__global__ void detectContactsCUDA(Contact*             contacts,
                                   const int            number_of_contacts,
                                   Cell*                cells,
                                   const int            number_of_cells,
                                   Face*                faces,
                                   Node*                nodes,
                                   InteractionProperty* interactions,
                                   const int            number_of_interactions)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_contacts; i += stride)
    {
        detectContact(contacts, i, cells, number_of_cells, faces, nodes, interactions, number_of_interactions);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     ____           _                  _           _               ____                   _                    _              ____   _   _   ____       _    
//    / ___|   __ _  | |   ___   _   _  | |   __ _  | |_    ___     / ___|   ___    _ __   | |_    __ _    ___  | |_   ___     / ___| | | | | |  _ \     / \   
//   | |      / _` | | |  / __| | | | | | |  / _` | | __|  / _ \   | |      / _ \  | '_ \  | __|  / _` |  / __| | __| / __|   | |     | | | | | | | |   / _ \  
//   | |___  | (_| | | | | (__  | |_| | | | | (_| | | |_  |  __/   | |___  | (_) | | | | | | |_  | (_| | | (__  | |_  \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \____|  \__,_| |_|  \___|  \__,_| |_|  \__,_|  \__|  \___|    \____|  \___/  |_| |_|  \__|  \__,_|  \___|  \__| |___/    \____|  \___/  |____/  /_/   \_\
//                                                                                                                                                             
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Calculates contacts on the GPU device
__global__ void calculateContactsCUDA(Contact*      contacts,
                                      const int     number_of_contacts,
                                      Face*         faces,
                                      Node*         nodes,
                                      const double  timestep)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_contacts; i += stride)
    {
        if (contacts[i].status == 1)
        {
            calculateContact(contacts, i, faces, nodes, timestep);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _                 _             _           _     ____                   _                            ____   _   _   ____       _    
//   | | | |  _ __     __| |   __ _  | |_    ___       / \    __  __ (_)   __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _   ___     / ___| | | | | |  _ \     / \   
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \     / _ \   \ \/ / | |  / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` | / __|   | |     | | | | | | | |   / _ \  
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/    / ___ \   >  <  | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| | \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   /_/   \_\ /_/\_\ |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, | |___/    \____|  \___/  |____/  /_/   \_\
//           |_|                                                                               |_|                         |___/                                           
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the axial springs on the GPU device
__global__ void updateAxialSpringsCUDA(AxialSpring*         axial_springs,
                                       const int		    number_of_springs,
                                       IntersectionNode*    intersection_nodes)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_springs; i += stride)
    {
        if (axial_springs[i].status == 1)
        {
            calculateAxialSpring(axial_springs, intersection_nodes, i);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _              ____            _             _     _                           _     ____                   _                            ____   _   _   ____       _    
//   | | | |  _ __     __| |   __ _  | |_    ___    |  _ \    ___   | |_    __ _  | |_  (_)   ___    _ __     __ _  | |   / ___|   _ __    _ __  (_)  _ __     __ _   ___     / ___| | | | | |  _ \     / \   
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \   | |_) |  / _ \  | __|  / _` | | __| | |  / _ \  | '_ \   / _` | | |   \___ \  | '_ \  | '__| | | | '_ \   / _` | / __|   | |     | | | | | | | |   / _ \  
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/   |  _ <  | (_) | | |_  | (_| | | |_  | | | (_) | | | | | | (_| | | |    ___) | | |_) | | |    | | | | | | | (_| | \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   |_| \_\  \___/   \__|  \__,_|  \__| |_|  \___/  |_| |_|  \__,_| |_|   |____/  | .__/  |_|    |_| |_| |_|  \__, | |___/    \____|  \___/  |____/  /_/   \_\
//           |_|                                                                                                                  |_|                         |___/                                           
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the rotational springs on the GPU device
__global__ void updateRotationalSpringsCUDA(RotationalSpring*   rotational_springs,
                                            const int		    number_of_rsprings,
                                            AxialSpring*        axial_springs,
                                            IntersectionNode*   intersection_nodes)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_rsprings; i += stride)
    {
        if (rotational_springs[i].status == 1)
        {
            calculateRotationalSpring(rotational_springs, axial_springs, intersection_nodes, i);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _              _   _               _                   ____   _   _   ____       _    
//   | | | |  _ __     __| |   __ _  | |_    ___    | \ | |   ___     __| |   ___   ___     / ___| | | | | |  _ \     / \   
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \   |  \| |  / _ \   / _` |  / _ \ / __|   | |     | | | | | | | |   / _ \  
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/   | |\  | | (_) | | (_| | |  __/ \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   |_| \_|  \___/   \__,_|  \___| |___/    \____|  \___/  |____/  /_/   \_\
//           |_|                                                                                                            
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the nodes on the GPU device
__global__ void updateNodesCUDA(Node*           nodes,
                                const int	    number_of_nodes,
                                const double	timestep,
                                const double    global_damping)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_nodes; i += stride)
    {
        integrateNode(nodes, i, timestep, global_damping);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _              ___           _                                       _     _                     _   _               _                   ____   _   _   ____       _    
//   | | | |  _ __     __| |   __ _  | |_    ___    |_ _|  _ __   | |_    ___   _ __   ___    ___    ___  | |_  (_)   ___    _ __     | \ | |   ___     __| |   ___   ___     / ___| | | | | |  _ \     / \   
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \    | |  | '_ \  | __|  / _ \ | '__| / __|  / _ \  / __| | __| | |  / _ \  | '_ \    |  \| |  / _ \   / _` |  / _ \ / __|   | |     | | | | | | | |   / _ \  
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/    | |  | | | | | |_  |  __/ | |    \__ \ |  __/ | (__  | |_  | | | (_) | | | | |   | |\  | | (_) | | (_| | |  __/ \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   |___| |_| |_|  \__|  \___| |_|    |___/  \___|  \___|  \__| |_|  \___/  |_| |_|   |_| \_|  \___/   \__,_|  \___| |___/    \____|  \___/  |____/  /_/   \_\
//           |_|                                                                                                                                                                                              
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the intersection nodes on the GPU device
__global__ void updateIntersectionNodesCUDA(IntersectionNode*   intersection_nodes,
                                            const int		    number_of_inodes,
                                            Node*               nodes)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_inodes; i += stride)
    {
        if (intersection_nodes[i].status == 1)
        {
            calculateIntersectionNodePosition(intersection_nodes, nodes, i);
            calculateIntersectionNodeVelocity(intersection_nodes, nodes, i);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _              _____                                  ____   _   _   ____       _    
//   | | | |  _ __     __| |   __ _  | |_    ___    |  ___|   __ _    ___    ___   ___     / ___| | | | | |  _ \     / \   
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \   | |_     / _` |  / __|  / _ \ / __|   | |     | | | | | | | |   / _ \  
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/   |  _|   | (_| | | (__  |  __/ \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|   |_|      \__,_|  \___|  \___| |___/    \____|  \___/  |____/  /_/   \_\
//           |_|                                                                                                           
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates the faces on the GPU device
__global__ void updateFacesCUDA(Face*       faces,
                                const int	number_of_faces,
                                Node*       nodes)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_faces; i += stride)
    {
        if (faces[i].status == 1)
        {
            calculateFaceCenter(faces, nodes, i);
            calculateFaceNormal(faces, nodes, i);
            calculateFaceArea(faces, nodes, i);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _   _               _           _               ____          _   _            ____   _   _   ____       _    
//   | | | |  _ __     __| |   __ _  | |_    ___     / ___|   ___  | | | |  ___     / ___| | | | | |  _ \     / \   
//   | | | | | '_ \   / _` |  / _` | | __|  / _ \   | |      / _ \ | | | | / __|   | |     | | | | | | | |   / _ \  
//   | |_| | | |_) | | (_| | | (_| | | |_  |  __/   | |___  |  __/ | | | | \__ \   | |___  | |_| | | |_| |  / ___ \ 
//    \___/  | .__/   \__,_|  \__,_|  \__|  \___|    \____|  \___| |_| |_| |___/    \____|  \___/  |____/  /_/   \_\
//           |_|                                                                                                    
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Updates cells on the GPU device
__global__ void updateCellsCUDA(Cell*               cells,
                                const int	        number_of_cells,
                                Node*               nodes,
                                IntersectionNode*   intersection_nodes,
                                Face*               faces,
                                AxialSpring*        axial_springs,
                                RotationalSpring*   rotational_springs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_cells; i += stride)
    {
        if (cells[i].status == 1)
        {
            calculateCellCenter(cells, nodes, i);
            calculateCellVolume(cells, nodes, i);
            checkCellDamage(cells, i, faces, intersection_nodes, axial_springs, rotational_springs);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    _____                          ____                   _                   __  __                                                  
//   |  ___|  _ __    ___    ___    |  _ \    ___  __   __ (_)   ___    ___    |  \/  |   ___   _ __ ___     ___     ___   _ __   _   _ 
//   | |_    | '__|  / _ \  / _ \   | | | |  / _ \ \ \ / / | |  / __|  / _ \   | |\/| |  / _ \ | '_ ` _ \   / _ \   / _ \ | '__| | | | |
//   |  _|   | |    |  __/ |  __/   | |_| | |  __/  \ V /  | | | (__  |  __/   | |  | | |  __/ | | | | | | | (_) | |  __/ | |    | |_| |
//   |_|     |_|     \___|  \___|   |____/   \___|   \_/   |_|  \___|  \___|   |_|  |_|  \___| |_| |_| |_|  \___/   \___| |_|     \__, |
//                                                                                                                                |___/ 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Frees the GPU memory
__host__ void freeDeviceMemory(void* dev_cells,
                               void* dev_faces,
                               void* dev_nodes,
                               void* dev_intersection_nodes,
                               void* dev_axial_springs,
                               void* dev_rotational_springs,
                               void* dev_contacts,
                               void* dev_interaction_properties)
{
    // Free the GPU memory
    cudaFree(dev_cells);
    cudaFree(dev_faces);
    cudaFree(dev_nodes);
    cudaFree(dev_intersection_nodes);
    cudaFree(dev_axial_springs);
    cudaFree(dev_rotational_springs);
    cudaFree(dev_contacts);
    cudaFree(dev_interaction_properties);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    __  __           _             ____    _                       _           _     _                 
//   |  \/  |   __ _  (_)  _ __     / ___|  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __  
//   | |\/| |  / _` | | | | '_ \    \___ \  | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \ 
//   | |  | | | (_| | | | | | | |    ___) | | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |
//   |_|  |_|  \__,_| |_| |_| |_|   |____/  |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|
//                                                                                                       
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{
    // Create the timers
    Timer global_timer;
    Timer local_timer;

    // Loading the output folder
    loadOutputFolder(settings);

    // Start the global timer
    global_timer.startTimer();

    // Create a variable for error codes
    int error = 0;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    ___          _   _     _          _   _               _     _                   ___   ___   _   _ 
    //   |_ _|  _ _   (_) | |_  (_)  __ _  | | (_)  ___  __ _  | |_  (_)  ___   _ _      / __| | _ \ | | | |
    //    | |  | ' \  | | |  _| | | / _` | | | | | |_ / / _` | |  _| | | / _ \ | ' \    | (__  |  _/ | |_| |
    //   |___| |_||_| |_|  \__| |_| \__,_| |_| |_| /__| \__,_|  \__| |_| \___/ |_||_|    \___| |_|    \___/ 
    //                                                                                                      
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    logData("### PROCESSING INPUT FILES ###", settings);
    local_timer.startTimer();
    error = processInputFiles(settings,
                              materials,
                              interaction_properties,
                              nodes,
                              cells);
    if (error != 0)
    {
        logData("ERROR: Processing the input files failed!", settings);
        return 1;
    }
    logData("Execution time of processing input files in milliseconds: " + std::to_string(local_timer.getDuration()), settings);


    local_timer.startTimer();
    logData("\n### CREATING SIMULATION COMPONENTS ###", settings);
    initializeSimulation(cells,
                         faces,
                         nodes,
                         intersection_nodes,
                         axial_springs,
                         rotational_springs,
                         external_forces,
                         contacts,
                         materials,
                         settings);
    logData("Execution time of creating simulation components in milliseconds: " + std::to_string(local_timer.getDuration()), settings);

    logData("\n### CHECKING SIMULATION COMPONENTS ###", settings);
    local_timer.startTimer();
    error = checkSimulation(cells,
                            faces,
                            nodes,
                            intersection_nodes,
                            axial_springs,
                            rotational_springs,
                            external_forces,
                            materials,
                            settings);
    if (error != 0)
    {
        logData("ERROR: Checking the simulation components failed!", settings);
        return 1;
    }
    logData("Execution time of checking simulation components in milliseconds: " + std::to_string(local_timer.getDuration()), settings);

    logData("\n### SAVING INITIAL CONFIGURATION ###", settings);
    local_timer.startTimer();
    int export_counter = 0;
    int step_counter = 0;
    int save_interval = int(settings.save_interval / settings.timestep);
    exportData(cells,
               faces,
               nodes,
               intersection_nodes,
               axial_springs,
               rotational_springs,
               contacts,
               external_forces,
               settings,
               export_counter);
    export_counter++;
    step_counter = 0;
    logData("Execution time of saving initial configuration in milliseconds: " + std::to_string(local_timer.getDuration()), settings);
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    ___          _   _     _          _   _               _     _                   ___   _   _   ___      _   
    //   |_ _|  _ _   (_) | |_  (_)  __ _  | | (_)  ___  __ _  | |_  (_)  ___   _ _      / __| | | | | |   \    /_\  
    //    | |  | ' \  | | |  _| | | / _` | | | | | |_ / / _` | |  _| | | / _ \ | ' \    | (__  | |_| | | |) |  / _ \ 
    //   |___| |_||_| |_|  \__| |_| \__,_| |_| |_| /__| \__,_|  \__| |_| \___/ |_||_|    \___|  \___/  |___/  /_/ \_\
    //                                                                                                               
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Creating variables for the GPU device
    const int number_of_cells = cells.size();
    const int number_of_faces = faces.size();
    const int number_of_nodes = nodes.size();
    const int number_of_intersection_nodes = intersection_nodes.size();
    const int number_of_axial_springs = axial_springs.size();
    const int number_of_rotational_springs = rotational_springs.size();
    const int number_of_contacts = contacts.size();
    const int number_of_interaction_properties = interaction_properties.size();

    const int number_of_blocks = settings.GPU_number_of_blocks;
    const int threads_per_block = settings.GPU_threads_per_block;

    cudaError_t cuda_status;

    // Check if we solve on the GPU device
    if (settings.simulation_on_GPU == 1)
    {
        logData("\n### INITIALIZING THE GPU DEVICE ###", settings);
        local_timer.startTimer();

        logData("Setting GPU device...", settings);
        cuda_status = setDevice(settings.GPU_device);
        if (cuda_status != cudaSuccess) return 1;

        logData("Allocate GPU device memory...", settings);
        cuda_status = allocateDeviceMemory((void**)&dev_cells, number_of_cells,
                                           (void**)&dev_faces, number_of_faces,
                                           (void**)&dev_nodes, number_of_nodes,
                                           (void**)&dev_intersection_nodes, number_of_intersection_nodes,
                                           (void**)&dev_axial_springs, number_of_axial_springs,
                                           (void**)&dev_rotational_springs, number_of_rotational_springs,
                                           (void**)&dev_contacts, number_of_contacts,
                                           (void**)&dev_interaction_properties, number_of_interaction_properties);
        if (cuda_status != cudaSuccess)
        {
            freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
            return 1;
        }

        logData("Copying data to GPU device...", settings);
        cuda_status = copyDataToDevice(dev_cells, &cells[0], number_of_cells,
                                       dev_faces, &faces[0], number_of_faces,
                                       dev_nodes, &nodes[0], number_of_nodes, 
                                       dev_intersection_nodes, &intersection_nodes[0], number_of_intersection_nodes, 
                                       dev_axial_springs, &axial_springs[0], number_of_axial_springs,
                                       dev_rotational_springs, &rotational_springs[0], number_of_rotational_springs,
                                       dev_contacts, &contacts[0], number_of_contacts,
                                       dev_interaction_properties, &interaction_properties[0], number_of_interaction_properties);
        if (cuda_status != cudaSuccess)
        {
            freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
            return 1;
        }
        logData("Execution time of initializing GPU device in milliseconds: " + std::to_string(local_timer.getDuration()), settings);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    ___                        _                   ___   _                  _          _     _              
    //   | _ \  _  _   _ _    _ _   (_)  _ _    __ _    / __| (_)  _ __    _  _  | |  __ _  | |_  (_)  ___   _ _  
    //   |   / | || | | ' \  | ' \  | | | ' \  / _` |   \__ \ | | | '  \  | || | | | / _` | |  _| | | / _ \ | ' \ 
    //   |_|_\  \_,_| |_||_| |_||_| |_| |_||_| \__, |   |___/ |_| |_|_|_|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_|
    //                                         |___/                                                              
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    logData("\n### RUNNING THE SIMULATION ###", settings);
    local_timer.startTimer();
    for (double simulation_time = settings.start_time; simulation_time <= settings.end_time; simulation_time += settings.timestep)
    {
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    ___                    _       _  _            _            
        //   | _ \  ___   ___  ___  | |_    | \| |  ___   __| |  ___   ___
        //   |   / / -_) (_-< / -_) |  _|   | .` | / _ \ / _` | / -_) (_-<
        //   |_|_\ \___| /__/ \___|  \__|   |_|\_| \___/ \__,_| \___| /__/
        //                                                                
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            resetNodesCUDA<<<number_of_blocks, threads_per_block>>>(dev_nodes, number_of_nodes);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) 
            {
                logData("resetNodesCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching resetNodesCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            resetNodesCPU(&nodes[0],
                nodes.size(),
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    ___          _                _        ___                _                 _        
        //   |   \   ___  | |_   ___   __  | |_     / __|  ___   _ _   | |_   __ _   __  | |_   ___
        //   | |) | / -_) |  _| / -_) / _| |  _|   | (__  / _ \ | ' \  |  _| / _` | / _| |  _| (_-<
        //   |___/  \___|  \__| \___| \__|  \__|    \___| \___/ |_||_|  \__| \__,_| \__|  \__| /__/
        //                                                                                         
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            detectContactsCUDA<<<number_of_blocks, threads_per_block>>>(dev_contacts, number_of_contacts, dev_cells, number_of_cells, dev_faces, dev_nodes, dev_interaction_properties, number_of_interaction_properties);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess)
            {
                logData("detectContactsCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching detectContactsCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            detectContactCPU(&contacts[0],
                contacts.size(),
                &cells[0],
                cells.size(),
                &faces[0],
                &nodes[0],
                &interaction_properties[0],
                interaction_properties.size(),
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //     ___          _               _          _              ___                _                 _        
        //    / __|  __ _  | |  __   _  _  | |  __ _  | |_   ___     / __|  ___   _ _   | |_   __ _   __  | |_   ___
        //   | (__  / _` | | | / _| | || | | | / _` | |  _| / -_)   | (__  / _ \ | ' \  |  _| / _` | / _| |  _| (_-<
        //    \___| \__,_| |_| \__|  \_,_| |_| \__,_|  \__| \___|    \___| \___/ |_||_|  \__| \__,_| \__|  \__| /__/
        //                                                                                                          
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            calculateContactsCUDA<<<number_of_blocks, threads_per_block >>>(dev_contacts, number_of_contacts, dev_faces, dev_nodes, settings.timestep);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess)
            {
                logData("calculateContactsCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching calculateContactsCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            calculateContactsCPU(&contacts[0],
                contacts.size(),
                &faces[0],
                &nodes[0],
                settings.timestep,
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    _   _             _          _               _           _          _     ___                _                    
        //   | | | |  _ __   __| |  __ _  | |_   ___      /_\   __ __ (_)  __ _  | |   / __|  _ __   _ _  (_)  _ _    __ _   ___
        //   | |_| | | '_ \ / _` | / _` | |  _| / -_)    / _ \  \ \ / | | / _` | | |   \__ \ | '_ \ | '_| | | | ' \  / _` | (_-<
        //    \___/  | .__/ \__,_| \__,_|  \__| \___|   /_/ \_\ /_\_\ |_| \__,_| |_|   |___/ | .__/ |_|   |_| |_||_| \__, | /__/
        //           |_|                                                                     |_|                     |___/      
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            updateAxialSpringsCUDA<<<number_of_blocks, threads_per_block >>>(dev_axial_springs, number_of_axial_springs, dev_intersection_nodes);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) 
            {
                logData("updateAxialSpringsCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching updateAxialSpringsCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            updateAxialSpringsCPU(&axial_springs[0],
                axial_springs.size(),
                &intersection_nodes[0],
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    _   _             _          _             ___         _            _     _                       _     ___                _                    
        //   | | | |  _ __   __| |  __ _  | |_   ___    | _ \  ___  | |_   __ _  | |_  (_)  ___   _ _    __ _  | |   / __|  _ __   _ _  (_)  _ _    __ _   ___
        //   | |_| | | '_ \ / _` | / _` | |  _| / -_)   |   / / _ \ |  _| / _` | |  _| | | / _ \ | ' \  / _` | | |   \__ \ | '_ \ | '_| | | | ' \  / _` | (_-<
        //    \___/  | .__/ \__,_| \__,_|  \__| \___|   |_|_\ \___/  \__| \__,_|  \__| |_| \___/ |_||_| \__,_| |_|   |___/ | .__/ |_|   |_| |_||_| \__, | /__/
        //           |_|                                                                                                   |_|                     |___/      
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            updateRotationalSpringsCUDA<<<number_of_blocks, threads_per_block >>>(dev_rotational_springs, number_of_rotational_springs, dev_axial_springs, dev_intersection_nodes);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess)
            {
                logData("updateRotationalSpringsCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess)
            {
                logData("After launching updateRotationalSpringsCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            updateRotationalSpringsCPU(&rotational_springs[0],
                rotational_springs.size(),
                &axial_springs[0],
                &intersection_nodes[0],
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //     ___                         ___           _              ___                         ___                _            
        //    / __|  ___   _ __   _  _    |   \   __ _  | |_   __ _    | __|  _ _   ___   _ __     |   \   ___  __ __ (_)  __   ___ 
        //   | (__  / _ \ | '_ \ | || |   | |) | / _` | |  _| / _` |   | _|  | '_| / _ \ | '  \    | |) | / -_) \ V / | | / _| / -_)
        //    \___| \___/ | .__/  \_, |   |___/  \__,_|  \__| \__,_|   |_|   |_|   \___/ |_|_|_|   |___/  \___|  \_/  |_| \__| \___|
        //                |_|     |__/                                                                                              
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            cuda_status = copyDataFromDevice(dev_cells, &cells[0], number_of_cells,
                dev_faces, &faces[0], number_of_faces,
                dev_nodes, &nodes[0], number_of_nodes,
                dev_intersection_nodes, &intersection_nodes[0], number_of_intersection_nodes,
                dev_axial_springs, &axial_springs[0], number_of_axial_springs,
                dev_rotational_springs, &rotational_springs[0], number_of_rotational_springs,
                dev_contacts, &contacts[0], number_of_contacts,
                dev_interaction_properties, &interaction_properties[0], number_of_interaction_properties);
            
            if (cuda_status != cudaSuccess)
            {
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //      _                   _              _           _          _     ___                _                   ___                             
        //     /_\    _ __   _ __  | |  _  _      /_\   __ __ (_)  __ _  | |   / __|  _ __   _ _  (_)  _ _    __ _    | __|  ___   _ _   __   ___   ___
        //    / _ \  | '_ \ | '_ \ | | | || |    / _ \  \ \ / | | / _` | | |   \__ \ | '_ \ | '_| | | | ' \  / _` |   | _|  / _ \ | '_| / _| / -_) (_-<
        //   /_/ \_\ | .__/ | .__/ |_|  \_, |   /_/ \_\ /_\_\ |_| \__,_| |_|   |___/ | .__/ |_|   |_| |_||_| \__, |   |_|   \___/ |_|   \__| \___| /__/
        //           |_|    |_|         |__/                                         |_|                     |___/                                     
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        applyAxialSpringForces(&axial_springs[0],
            axial_springs.size(),
            &intersection_nodes[0],
            &nodes[0]);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //      _                   _            ___         _            _     _                       _     ___                _                   ___                             
        //     /_\    _ __   _ __  | |  _  _    | _ \  ___  | |_   __ _  | |_  (_)  ___   _ _    __ _  | |   / __|  _ __   _ _  (_)  _ _    __ _    | __|  ___   _ _   __   ___   ___
        //    / _ \  | '_ \ | '_ \ | | | || |   |   / / _ \ |  _| / _` | |  _| | | / _ \ | ' \  / _` | | |   \__ \ | '_ \ | '_| | | | ' \  / _` |   | _|  / _ \ | '_| / _| / -_) (_-<
        //   /_/ \_\ | .__/ | .__/ |_|  \_, |   |_|_\ \___/  \__| \__,_|  \__| |_| \___/ |_||_| \__,_| |_|   |___/ | .__/ |_|   |_| |_||_| \__, |   |_|   \___/ |_|   \__| \___| /__/
        //           |_|    |_|         |__/                                                                       |_|                     |___/                                     
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        applyRotationalSpringForces(&rotational_springs[0],
            rotational_springs.size(),
            &axial_springs[0],
            &intersection_nodes[0],
            &nodes[0]);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //      _                   _             ___                _                 _       ___                             
        //     /_\    _ __   _ __  | |  _  _     / __|  ___   _ _   | |_   __ _   __  | |_    | __|  ___   _ _   __   ___   ___
        //    / _ \  | '_ \ | '_ \ | | | || |   | (__  / _ \ | ' \  |  _| / _` | / _| |  _|   | _|  / _ \ | '_| / _| / -_) (_-<
        //   /_/ \_\ | .__/ | .__/ |_|  \_, |    \___| \___/ |_||_|  \__| \__,_| \__|  \__|   |_|   \___/ |_|   \__| \___| /__/
        //           |_|    |_|         |__/                                                                                   
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        applyContactForces(&contacts[0],
            contacts.size(),
            &faces[0],
            &nodes[0]);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //      _                   _            ___         _                               _     ___                             
        //     /_\    _ __   _ __  | |  _  _    | __| __ __ | |_   ___   _ _   _ _    __ _  | |   | __|  ___   _ _   __   ___   ___
        //    / _ \  | '_ \ | '_ \ | | | || |   | _|  \ \ / |  _| / -_) | '_| | ' \  / _` | | |   | _|  / _ \ | '_| / _| / -_) (_-<
        //   /_/ \_\ | .__/ | .__/ |_|  \_, |   |___| /_\_\  \__| \___| |_|   |_||_| \__,_| |_|   |_|   \___/ |_|   \__| \___| /__/
        //           |_|    |_|         |__/                                                                                       
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if there is any external force defined
        if (external_forces.size() > 0)
        {
            applyExternalForces(&external_forces[0],
                external_forces.size(),
                &nodes[0],
                simulation_time);
        }
 
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //     ___                         ___           _              _____           ___                _            
        //    / __|  ___   _ __   _  _    |   \   __ _  | |_   __ _    |_   _|  ___    |   \   ___  __ __ (_)  __   ___ 
        //   | (__  / _ \ | '_ \ | || |   | |) | / _` | |  _| / _` |     | |   / _ \   | |) | / -_) \ V / | | / _| / -_)
        //    \___| \___/ | .__/  \_, |   |___/  \__,_|  \__| \__,_|     |_|   \___/   |___/  \___|  \_/  |_| \__| \___|
        //                |_|     |__/                                                                                  
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            cuda_status = copyDataToDevice(dev_cells, &cells[0], number_of_cells,
                dev_faces, &faces[0], number_of_faces,
                dev_nodes, &nodes[0], number_of_nodes,
                dev_intersection_nodes, &intersection_nodes[0], number_of_intersection_nodes,
                dev_axial_springs, &axial_springs[0], number_of_axial_springs,
                dev_rotational_springs, &rotational_springs[0], number_of_rotational_springs,
                dev_contacts, &contacts[0], number_of_contacts,
                dev_interaction_properties, &interaction_properties[0], number_of_interaction_properties);
            if (cuda_status != cudaSuccess)
            {
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    _   _             _          _             _  _            _            
        //   | | | |  _ __   __| |  __ _  | |_   ___    | \| |  ___   __| |  ___   ___
        //   | |_| | | '_ \ / _` | / _` | |  _| / -_)   | .` | / _ \ / _` | / -_) (_-<
        //    \___/  | .__/ \__,_| \__,_|  \__| \___|   |_|\_| \___/ \__,_| \___| /__/
        //           |_|                                                              
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            updateNodesCUDA <<<number_of_blocks, threads_per_block >>> (dev_nodes, number_of_nodes, settings.timestep, settings.global_damping);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) 
            {
                logData("updateNodesCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching updateNodesCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            updateNodesCPU(&nodes[0],
                nodes.size(),
                settings.number_of_CPU_threads,
                settings.timestep,
                settings.global_damping);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    _   _             _          _             ___          _                                 _     _                  _  _            _            
        //   | | | |  _ __   __| |  __ _  | |_   ___    |_ _|  _ _   | |_   ___   _ _   ___  ___   __  | |_  (_)  ___   _ _     | \| |  ___   __| |  ___   ___
        //   | |_| | | '_ \ / _` | / _` | |  _| / -_)    | |  | ' \  |  _| / -_) | '_| (_-< / -_) / _| |  _| | | / _ \ | ' \    | .` | / _ \ / _` | / -_) (_-<
        //    \___/  | .__/ \__,_| \__,_|  \__| \___|   |___| |_||_|  \__| \___| |_|   /__/ \___| \__|  \__| |_| \___/ |_||_|   |_|\_| \___/ \__,_| \___| /__/
        //           |_|                                                                                                                                      
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            updateIntersectionNodesCUDA <<<number_of_blocks, threads_per_block >>> (dev_intersection_nodes, number_of_intersection_nodes, dev_nodes);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) 
            {
                logData("updateIntersectionNodesCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching updateIntersectionNodesCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            updateIntersectionNodesCPU(&intersection_nodes[0],
                intersection_nodes.size(),
                &nodes[0],
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    _   _             _          _             ___                        
        //   | | | |  _ __   __| |  __ _  | |_   ___    | __|  __ _   __   ___   ___
        //   | |_| | | '_ \ / _` | / _` | |  _| / -_)   | _|  / _` | / _| / -_) (_-<
        //    \___/  | .__/ \__,_| \__,_|  \__| \___|   |_|   \__,_| \__| \___| /__/
        //           |_|                                                            
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            updateFacesCUDA <<<number_of_blocks, threads_per_block >>> (dev_faces, number_of_faces, dev_nodes);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) 
            {
                logData("updateFacesCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching updateFacesCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            updateFacesCPU(&faces[0],
                faces.size(),
                &nodes[0],
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    _   _             _          _              ___         _   _      
        //   | | | |  _ __   __| |  __ _  | |_   ___     / __|  ___  | | | |  ___
        //   | |_| | | '_ \ / _` | / _` | |  _| / -_)   | (__  / -_) | | | | (_-<
        //    \___/  | .__/ \__,_| \__,_|  \__| \___|    \___| \___| |_| |_| /__/
        //           |_|                                                         
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if we solve on the device
        if (settings.simulation_on_GPU == 1)
        {
            // Launch the CUDA kernel
            updateCellsCUDA <<<number_of_blocks, threads_per_block >>> (dev_cells, number_of_cells, dev_nodes, dev_intersection_nodes, dev_faces, dev_axial_springs, dev_rotational_springs);

            // Check for any errors launching the kernel
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) 
            {
                logData("updateCellsCUDA launch failed: " + std::string(cudaGetErrorString(cuda_status)), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }

            // Waiting for the kernel to finish
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) 
            {
                logData("After launching updateCellsCUDA cudaDeviceSynchronize returned error code: " + std::to_string(cuda_status), settings);
                freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                return 1;
            }
        }
        else
        {
            // Launch the CPU kernel
            updateCellsCPU(&cells[0],
                cells.size(),
                &nodes[0],
                &intersection_nodes[0],
                &faces[0],
                &axial_springs[0],
                &rotational_springs[0],
                settings.number_of_CPU_threads);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //    ___                _               
        //   / __|  __ _  __ __ (_)  _ _    __ _ 
        //   \__ \ / _` | \ V / | | | ' \  / _` |
        //   |___/ \__,_|  \_/  |_| |_||_| \__, |
        //                                 |___/ 
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Increase the step counter
        step_counter++;

        // Check if the step counter and the target save interval are equal
        if (step_counter == save_interval)
        {
            // Send some messages to the user about the course of the simulation
            logData("\nSimulation is " + std::to_string((simulation_time / settings.end_time) * 100) + " % completed", settings);
            logData("Execution time of " + std::to_string(save_interval) + " iterations in milliseconds : " + std::to_string(local_timer.getDuration()), settings);
            logData("Saving at time " + std::to_string(simulation_time) + " second", settings);

            // Start saving the data
            local_timer.startTimer();

            // Copy data back from the device if we solve on the GPU
            if (settings.simulation_on_GPU == 1)
            {
                cuda_status = copyDataFromDevice(dev_cells, &cells[0], number_of_cells,
                                                 dev_faces, &faces[0], number_of_faces,
                                                 dev_nodes, &nodes[0], number_of_nodes,
                                                 dev_intersection_nodes, &intersection_nodes[0], number_of_intersection_nodes,
                                                 dev_axial_springs, &axial_springs[0], number_of_axial_springs,
                                                 dev_rotational_springs, &rotational_springs[0], number_of_rotational_springs,
                                                 dev_contacts, &contacts[0], number_of_contacts,
                                                 dev_interaction_properties, &interaction_properties[0], number_of_interaction_properties);

                if (cuda_status != cudaSuccess)
                {
                    freeDeviceMemory(dev_cells, dev_faces, dev_nodes, dev_intersection_nodes, dev_axial_springs, dev_rotational_springs, dev_contacts, dev_interaction_properties);
                    return 1;
                }
            }

            // Export the data
            exportData(cells,
                       faces,
                       nodes,
                       intersection_nodes,
                       axial_springs,
                       rotational_springs,
                       contacts,
                       external_forces,
                       settings,
                       export_counter);
            
            // Increase the export counter and reset the step counter
            export_counter++;
            step_counter = 0;

            logData("Execution time of the save in milliseconds: " + std::to_string(local_timer.getDuration()), settings);

            // Start the timer for the next round of iterations
            local_timer.startTimer();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    ___   _          _        _      _                   ___   _                  _          _     _              
    //   | __| (_)  _ _   (_)  ___ | |_   (_)  _ _    __ _    / __| (_)  _ __    _  _  | |  __ _  | |_  (_)  ___   _ _  
    //   | _|  | | | ' \  | | (_-< | ' \  | | | ' \  / _` |   \__ \ | | | '  \  | || | | | / _` | |  _| | | / _ \ | ' \ 
    //   |_|   |_| |_||_| |_| /__/ |_||_| |_| |_||_| \__, |   |___/ |_| |_|_|_|  \_,_| |_| \__,_|  \__| |_| \___/ |_||_|
    //                                               |___/                                                              
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    logData("\n### SAVING THE FINAL CONFIGURATION ###", settings);
    local_timer.startTimer();
    exportData(cells,
               faces,
               nodes,
               intersection_nodes,
               axial_springs,
               rotational_springs,
               contacts,
               external_forces,
               settings,
               export_counter);
    logData("Execution time of the final configuration in milliseconds: " + std::to_string(local_timer.getDuration()), settings);

     // Cuda device reset must be called before exiting
    if (settings.simulation_on_GPU == 1)
    {
        cudaError_t cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) 
        {
            logData("cudaDeviceReset failed!", settings);
            return 1;
        }
    }

    logData("\n### SIMULATION COMPLETED ###", settings);
    logData("Total simulation time in milliseconds: " + std::to_string(global_timer.getDuration()), settings);

    return 0;
}
