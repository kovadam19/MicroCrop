#include "Simulation.h"


//    ____            _                          _   
//   |  _ \    __ _  | |_    __ _   ___    ___  | |_ 
//   | | | |  / _` | | __|  / _` | / __|  / _ \ | __|
//   | |_| | | (_| | | |_  | (_| | \__ \ |  __/ | |_ 
//   |____/   \__,_|  \__|  \__,_| |___/  \___|  \__|
//                                                   

Settings settings;
MaterialContainer materials;
NodeContainer nodes;
IntersectionNodeContainer intersection_nodes;
FaceContainer faces;
CellContainer cells;
AxialSpringContainer axial_springs;
RotationalSpringContainer rotational_springs;
ExternalForceContainer external_forces;
ContactContainer contacts;


int main()
{

    std::cout << "### INITIALIZATION ###" << std::endl;
    std::cout << "Processing input files..." << std::endl;
    processInputFiles(settings,
        materials,
        nodes,
        cells);
    std::cout << "Initializing the simulation..." << std::endl;
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

    std::cout << "### CHECKING SIMULATION ###" << std::endl;
    checkSimulation(cells,
        faces,
        nodes,
        intersection_nodes,
        axial_springs,
        rotational_springs,
        external_forces,
        materials,
        settings);

    if (settings.simulation_on_GPU == 1)
    {
        std::cout << "### RUNNING SIMULATION ON GPU###" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        runSimulationCUDA(cells,
            faces,
            nodes,
            intersection_nodes,
            axial_springs,
            rotational_springs,
            external_forces,
            settings);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << "Simulation execution time on GPU in seconds: " << duration.count() << std::endl;
    }
    else
    {
        std::cout << "### RUNNING SIMULATION ON CPU###" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        runSimulationCPU(cells,
            faces,
            nodes,
            intersection_nodes,
            axial_springs,
            rotational_springs,
            external_forces,
            settings);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << "Simulation execution time on CPU in seconds: " << duration.count() << std::endl;
    }








    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
