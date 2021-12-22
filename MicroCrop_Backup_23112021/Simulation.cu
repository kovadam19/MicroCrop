#include "Simulation.h"


__global__ void resetNodesCUDA(Node* nodes,
                               int	 number_of_nodes)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_nodes; i += stride)
    {
        nodes[i].force = make_double3(0.0, 0.0, 0.0);
    }
}


__global__ void updateAxialSpringsCUDA(AxialSpring* axial_springs,
    int		number_of_springs,
    IntersectionNode* intersection_nodes)
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

//__global__ void applyAxialSpringForcesCUDA(AxialSpring* axial_springs,
//    int		number_of_springs,
//    IntersectionNode* intersection_nodes,
//    Node* nodes)
//{
//    applyAxialSpringForces(axial_springs, number_of_springs, intersection_nodes, nodes);
//}

__global__ void updateRotationalSpringsCUDA(RotationalSpring* rotational_springs,
    int		number_of_rsprings,
    AxialSpring* axial_springs,
    IntersectionNode* intersection_nodes)
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

//__global__ void applyRotationalSpringForcesCUDA(RotationalSpring* rotational_springs,
//    int		number_of_rsprings,
//    AxialSpring* axial_springs,
//    IntersectionNode* intersection_nodes,
//    Node* nodes)
//{
//
//    applyRotationalSpringForces(rotational_springs, number_of_rsprings, axial_springs, intersection_nodes, nodes);
//}

//__global__ void applyExternalForcesCUDA(ExternalForce* external_forces,
//    int	number_of_forces,
//    Node* nodes,
//    double			time)
//{
//    applyExternalForces(external_forces, number_of_forces, nodes, time);
//}

__global__ void updateNodesCUDA(Node* nodes,
    int	number_of_nodes,
    double			timestep)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_nodes; i += stride)
    {
        integrateNode(nodes, i, timestep);
    }
}

__global__ void updateIntersectionNodesCUDA(IntersectionNode* intersection_nodes,
    int		number_of_inodes,
    Node* nodes)
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

__global__ void updateFacesCUDA(Face* faces,
    int	number_of_faces,
    Node* nodes)
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

__global__ void updateCellsCUDA(Cell* cells,
    int	number_of_cells,
    Node* nodes,
    IntersectionNode* intersection_nodes,
    Face* faces,
    AxialSpring* axial_springs,
    RotationalSpring* rotational_springs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number_of_cells; i += stride)
    {
        if (cells[i].status == 1)
        {
            calculateCellCenter(cells, nodes, i);
            calculateCellVolume(cells, nodes, i);
            calculateCellCircumsphere(cells, nodes, i);
            checkCellDamage(cells, i, faces, intersection_nodes, axial_springs, rotational_springs);
        }
    }
}



__host__ cudaError_t runSimulationCUDA(CellContainer& host_cells,
    FaceContainer& host_faces,
    NodeContainer& host_nodes,
    IntersectionNodeContainer& host_intersection_nodes,
    AxialSpringContainer& host_axial_springs,
    RotationalSpringContainer& host_rotational_springs,
    ExternalForceContainer& host_external_forces,
    Settings& host_settings)
{
    // Variables for testing performance
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    
    std::cout << "Initializing the GPU device..." << std::endl;

    // Initializing the device variables
    Cell* dev_cells;
    const int number_of_cells = host_cells.size();
    Face* dev_faces;
    const int number_of_faces = host_faces.size();
    Node* dev_nodes;
    const int number_of_nodes = host_nodes.size();
    IntersectionNode* dev_intersection_nodes;
    const int number_of_intersection_nodes = host_intersection_nodes.size();
    AxialSpring* dev_axial_springs;
    const int number_of_axial_springs = host_axial_springs.size();
    RotationalSpring* dev_rotational_springs;
    const int number_of_rotational_springs = host_rotational_springs.size();
    //ExternalForce* dev_external_forces;
    //const int number_of_external_forces = host_external_forces.size();
    cudaError_t cudaStatus;

    // Initialize the device
    cudaStatus = cudaSetDevice(host_settings.GPU_device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    // Allocate GPU buffers for device containers
    cudaStatus = cudaMalloc((void**)&dev_cells, number_of_cells * sizeof(Cell));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_cells!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_faces, number_of_faces * sizeof(Face));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_faces!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nodes, number_of_nodes * sizeof(Node));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_nodes!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_intersection_nodes, number_of_intersection_nodes * sizeof(IntersectionNode));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_intersection_nodes!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_axial_springs, number_of_axial_springs * sizeof(AxialSpring));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_axial_springs!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rotational_springs, number_of_rotational_springs * sizeof(RotationalSpring));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_rotational_springs!");
        goto Error;
    }

    //cudaStatus = cudaMalloc((void**)&dev_external_forces, number_of_external_forces * sizeof(ExternalForce));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed on dev_external_forces!");
    //    goto Error;
    //}

    // Copy containers from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_cells, &host_cells[0], number_of_cells * sizeof(Cell), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_cells!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_faces, &host_faces[0], number_of_faces * sizeof(Face), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_faces!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_nodes, &host_nodes[0], number_of_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_nodes!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_intersection_nodes, &host_intersection_nodes[0], number_of_intersection_nodes * sizeof(IntersectionNode), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_intersection_nodes!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_axial_springs, &host_axial_springs[0], number_of_axial_springs * sizeof(AxialSpring), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_axial_springs!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rotational_springs, &host_rotational_springs[0], number_of_rotational_springs * sizeof(RotationalSpring), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_rotational_springs!");
        goto Error;
    }

    //cudaStatus = cudaMemcpy(dev_external_forces, &host_external_forces[0], number_of_external_forces * sizeof(ExternalForce), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed on dev_external_forces!");
    //    goto Error;
    //}

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Execution time of GPU initialization in milliseconds: " << duration.count() << std::endl;


    std::cout << "Starting calculation on the GPU device..." << std::endl;
    start = std::chrono::high_resolution_clock::now();


    double simulation_time = host_settings.start_time;
    double simulation_end_time = host_settings.end_time;
    int save_interval = int(host_settings.save_interval / host_settings.timestep);
    int step_counter = 0;
    int export_counter = 0;

    int threads_per_block = host_settings.GPU_threads_per_block;
    int number_of_blocks = host_settings.GPU_number_of_blocks;


    while (simulation_time <= simulation_end_time)
    {
        if (step_counter == save_interval)
        {
            if (simulation_time != host_settings.start_time)
            {
                stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "Execution time of " << save_interval << " iterations in milliseconds: " << duration.count() << std::endl;
            }

            std::cout << "Simulation is " << (simulation_time / simulation_end_time) * 100 << " % completed." << std::endl;

            // Saving
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Saving at time " << simulation_time << " second." << std::endl;

            // Copy containers from GPU buffer to host memory
            if (host_settings.save_nodes == 1 || host_settings.save_faces == 1 || host_settings.save_cells == 1)
            {
                cudaStatus = cudaMemcpy(&host_nodes[0], dev_nodes, number_of_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed on host_nodes!");
                    goto Error;
                }
            }
    
            if (host_settings.save_faces == 1)
            {
                cudaStatus = cudaMemcpy(&host_faces[0], dev_faces, number_of_faces * sizeof(Face), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed on host_faces!");
                    goto Error;
                }
            }

            if (host_settings.save_cells == 1)
            {
                cudaStatus = cudaMemcpy(&host_cells[0], dev_cells, number_of_cells * sizeof(Cell), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed on host_cells!");
                    goto Error;
                }
            }

            if (host_settings.save_axial_springs == 1 || host_settings.save_rotational_springs == 1)
            {
                cudaStatus = cudaMemcpy(&host_intersection_nodes[0], dev_intersection_nodes, number_of_intersection_nodes * sizeof(IntersectionNode), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed on host_intersection_nodes!");
                    goto Error;
                }
            }

            if (host_settings.save_axial_springs == 1 || host_settings.save_rotational_springs == 1)
            {
                cudaStatus = cudaMemcpy(&host_axial_springs[0], dev_axial_springs, number_of_axial_springs * sizeof(AxialSpring), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed on host_axial_springs!");
                    goto Error;
                }
            }

            if (host_settings.save_rotational_springs == 1)
            {
                cudaStatus = cudaMemcpy(&host_rotational_springs[0], dev_rotational_springs, number_of_rotational_springs * sizeof(RotationalSpring), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed on host_rotational_springs!");
                    goto Error;
                }
            }

            exportSimulationData(host_cells,
                host_faces,
                host_nodes,
                host_intersection_nodes,
                host_axial_springs,
                host_rotational_springs,
                host_settings,
                export_counter);

            export_counter++;
            step_counter = 0;

            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Execution time of the save in milliseconds: " << duration.count() << std::endl;

            start = std::chrono::high_resolution_clock::now();
        }







        start = std::chrono::high_resolution_clock::now();
        resetNodesCUDA<<<number_of_blocks, threads_per_block>>>(dev_nodes, number_of_nodes);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "resetNodesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching resetNodesCUDA!\n", cudaStatus);
            goto Error;
        }

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of resetNodesCUDA in milliseconds: " << duration.count() << std::endl;







        start = std::chrono::high_resolution_clock::now();
        updateAxialSpringsCUDA<<<number_of_blocks, threads_per_block >>>(dev_axial_springs, number_of_axial_springs, dev_intersection_nodes);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updateAxialSpringsCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateAxialSpringsCUDA!\n", cudaStatus);
            goto Error;
        }

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of updateAxialSpringsCUDA in milliseconds: " << duration.count() << std::endl;







        //start = std::chrono::high_resolution_clock::now();
        //applyAxialSpringForcesCUDA<<<1, 1>>>(dev_axial_springs, number_of_axial_springs, dev_intersection_nodes, dev_nodes);

        //// Check for any errors launching the kernel
        //cudaStatus = cudaGetLastError();
        //if (cudaStatus != cudaSuccess) {
        //    fprintf(stderr, "applyAxialSpringForcesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //    goto Error;
        //}

        //// Waiting for the kernel to finish
        //cudaStatus = cudaDeviceSynchronize();
        //if (cudaStatus != cudaSuccess) {
        //    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching applyAxialSpringForcesCUDA!\n", cudaStatus);
        //    goto Error;
        //}

        //stop = std::chrono::high_resolution_clock::now();
        //duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        //std::cout << "Execution time of applyAxialSpringForcesCUDA in milliseconds: " << duration.count() << std::endl;








        start = std::chrono::high_resolution_clock::now();
        updateRotationalSpringsCUDA<<<number_of_blocks, threads_per_block >>>(dev_rotational_springs, number_of_rotational_springs, dev_axial_springs, dev_intersection_nodes);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updateRotationalSpringsCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateRotationalSpringsCUDA!\n", cudaStatus);
            goto Error;
        }

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of updateRotationalSpringsCUDA in milliseconds: " << duration.count() << std::endl;






        //start = std::chrono::high_resolution_clock::now();
        //applyRotationalSpringForcesCUDA<<<1, 1 >>>(dev_rotational_springs, number_of_rotational_springs, dev_axial_springs, dev_intersection_nodes, dev_nodes);

        //// Check for any errors launching the kernel
        //cudaStatus = cudaGetLastError();
        //if (cudaStatus != cudaSuccess) {
        //    fprintf(stderr, "applyRotationalSpringForcesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //    goto Error;
        //}

        //// Waiting for the kernel to finish
        //cudaStatus = cudaDeviceSynchronize();
        //if (cudaStatus != cudaSuccess) {
        //    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching applyRotationalSpringForcesCUDA!\n", cudaStatus);
        //    goto Error;
        //}

        //stop = std::chrono::high_resolution_clock::now();
        //duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        //std::cout << "Execution time of applyRotationalSpringForcesCUDA in milliseconds: " << duration.count() << std::endl;






        //start = std::chrono::high_resolution_clock::now();
        //applyExternalForcesCUDA<<<1, 1 >>>(dev_external_forces, number_of_external_forces, dev_nodes, simulation_time);

        //// Check for any errors launching the kernel
        //cudaStatus = cudaGetLastError();
        //if (cudaStatus != cudaSuccess) {
        //    fprintf(stderr, "applyExternalForcesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //    goto Error;
        //}

        //// Waiting for the kernel to finish
        //cudaStatus = cudaDeviceSynchronize();
        //if (cudaStatus != cudaSuccess) {
        //    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching applyExternalForcesCUDA!\n", cudaStatus);
        //    goto Error;
        //}

        //stop = std::chrono::high_resolution_clock::now();
        //duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        //std::cout << "Execution time of applyExternalForcesCUDA in milliseconds: " << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        cudaStatus = cudaMemcpy(&host_nodes[0], dev_nodes, number_of_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on host_nodes!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(&host_intersection_nodes[0], dev_intersection_nodes, number_of_intersection_nodes * sizeof(IntersectionNode), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on host_intersection_nodes!");
            goto Error;
        }
       
        cudaStatus = cudaMemcpy(&host_axial_springs[0], dev_axial_springs, number_of_axial_springs * sizeof(AxialSpring), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on host_axial_springs!");
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(&host_rotational_springs[0], dev_rotational_springs, number_of_rotational_springs * sizeof(RotationalSpring), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on host_rotational_springs!");
            goto Error;
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of copying data from device to host to apply forces in milliseconds: " << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        applyAxialSpringForces(&host_axial_springs[0],
            number_of_axial_springs,
            &host_intersection_nodes[0],
            &host_nodes[0]);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of applyAxialSpringForces in milliseconds: " << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        applyRotationalSpringForces(&host_rotational_springs[0],
            number_of_rotational_springs,
            &host_axial_springs[0],
            &host_intersection_nodes[0],
            &host_nodes[0]);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of applyRotationalSpringForces in milliseconds: " << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        applyExternalForces(&host_external_forces[0],
            host_external_forces.size(),
            &host_nodes[0],
            simulation_time);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of applyExternalForces in milliseconds: " << duration.count() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        cudaStatus = cudaMemcpy(dev_nodes, &host_nodes[0], number_of_nodes * sizeof(Node), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on dev_nodes!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_intersection_nodes, &host_intersection_nodes[0], number_of_intersection_nodes * sizeof(IntersectionNode), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on dev_intersection_nodes!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_axial_springs, &host_axial_springs[0], number_of_axial_springs * sizeof(AxialSpring), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on dev_axial_springs!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_rotational_springs, &host_rotational_springs[0], number_of_rotational_springs * sizeof(RotationalSpring), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed on dev_rotational_springs!");
            goto Error;
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of copying data from host to device after applying forces in milliseconds: " << duration.count() << std::endl;



        start = std::chrono::high_resolution_clock::now();
        updateNodesCUDA <<<number_of_blocks, threads_per_block >>> (dev_nodes, number_of_nodes, host_settings.timestep);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updateNodesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateNodesCUDA!\n", cudaStatus);
            goto Error;
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of updateNodesCUDA in milliseconds: " << duration.count() << std::endl;







        start = std::chrono::high_resolution_clock::now();
        updateIntersectionNodesCUDA <<<number_of_blocks, threads_per_block >>> (dev_intersection_nodes, number_of_intersection_nodes, dev_nodes);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updateIntersectionNodesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateIntersectionNodesCUDA!\n", cudaStatus);
            goto Error;
        }

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of updateIntersectionNodesCUDA in milliseconds: " << duration.count() << std::endl;





        start = std::chrono::high_resolution_clock::now();
        updateFacesCUDA <<<number_of_blocks, threads_per_block >>> (dev_faces, number_of_faces, dev_nodes);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updateFacesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateFacesCUDA!\n", cudaStatus);
            goto Error;
        }

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of updateFacesCUDA in milliseconds: " << duration.count() << std::endl;






        start = std::chrono::high_resolution_clock::now();
        updateCellsCUDA <<<number_of_blocks, threads_per_block >>> (dev_cells, number_of_cells, dev_nodes, dev_intersection_nodes, dev_faces, dev_axial_springs, dev_rotational_springs);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "updateCellsCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Waiting for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateCellsCUDA!\n", cudaStatus);
            goto Error;
        }

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time of updateCellsCUDA in milliseconds: " << duration.count() << std::endl;




        step_counter++;
        simulation_time += host_settings.timestep;
    }


Error:
    // Free the GPU memory
    cudaFree(dev_cells);
    cudaFree(dev_faces);
    cudaFree(dev_nodes);
    cudaFree(dev_intersection_nodes);
    cudaFree(dev_axial_springs);
    cudaFree(dev_rotational_springs);
    //cudaFree(dev_external_forces);

    return cudaStatus;
}



__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
