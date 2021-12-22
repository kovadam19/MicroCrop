# MicroCrop - Crop Solver by Adam

## Motivation

This application was created to practice C++ & CUDA C/C++ coding after completing [NVIDIA Fundamentals of Accelerated Computing with CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/about).

Since the beginning of my university studies of agricultural engineering, simulating agricultural equipment is my passion. 
I was especially interested in new technologies to simulate processing of agricultural materials. 
One day I ran into an interesting study on mass-spring systems: [Oussama Jarrousse: Modified Mass-Spring System for Physically Based Deformation Modeling](https://www.researchgate.net/publication/342899408_Modified_Mass-spring_System_for_Physically_Based_Deformation_Modeling). 
The author used this method to simulate human organs like breast and heart, so I thought it can be used for agricultural materials too.
Nonetheless, implementing this method into a C++ & CUDA C/C++ code seemed to be a perfect exercise to practice my new knowledge.

My certificate:

![Certificate](documentation/Certificate.JPG "Certificate")

## Short description

In the simulation, each component (body) is consisted of tetrahedral cells.
The mechanical behaviour of the cells are defined by axial and rotational springs and dampers.
Contact is detected among the cells that belong to the same or different components.
The mechanical behaviour of the contact between two components is determined by the interaction properties such as contact stiffness and coefficient of static friction.
The components are able to deform according to the contact and external forces acting on them.
The cells can be damaged and this can result in breakage of the component.
Initial and boundary conditions can be defined onto the components at the beginning of the simulation.
Nonetheless, external forces (like gravity) and fixed velocities can be applied onto the components as well.

Input parameters:
* Tetrahedral mesh for each component;
* Material definition for each component;
* Interaction definition among the components;
* Initial conditions;
* Boundary conditions;
* External forces;
* Fixed velocities;
* Solver settings;
* Save setting.

Result files:
* Log file (TXT) with solver messages;
* Node output file (VTK);
* Cell output file (VTK);
* Face output file (VTK);
* Axial spring output file (VTK);
* Rotational spring output file (VTK);
* Contact output file (VTK);
* External force output file (VTK).

The VTK output files can be processed and visualized in [Paraview](https://www.paraview.org/).

The solver was developed in [Microsoft Visual Studio 2019](https://visualstudio.microsoft.com/) by using [ISO C++14 standard](https://en.cppreference.com/w/cpp/14) and [NVIDIA CUDA 11.3](https://docs.nvidia.com/cuda/index.html).

![QuickOverView](documentation/QuickOverview.png "Quick Overview")

## User guide

In this section the usage of the application will be presented.

### Input files

The input parameters for the solver are defined in multiple files. 
Let's start with the basic Settings.txt file.

#### Settings

The Settings.txt has to be located in the root **/0_INPUT** folder.
In the Settings.txt file there are multiple keywords defined to control the input paramters.
Some of them is optional and some of them is mandatory.
On the following list the **mandatory** keywords are **bold**:
* **NUMBER_OF_COMPONENTS** *numComponents*: Defines the number of components (bodies) in the simulation. This has to be at the very top of the setting file.
* **CELLS** *numComponents*: Defines the location of the tetrahedral mesh files. Each component has to have a mesh file.
* **MATERIALS** *numComponents*: Defines the location of the material definition files. Each component has to have a material file.

* INTERACTIONS *numInteractions*: Defines the location of the interaction file(s).
* INITIAL_CONDITION *numInitialCondition*: Defines the location of the initial condition file(s).
* BOUNDARY_CONDITION *numBoundaryCondition*: Defines the location of the boundary condition file(s).
* EXTERNAL_FORCE *numExternalForce*: Defines the location of the external force file(s).
* FIXED_VELOCITY *numFixedVelocity*: Defines the location of the fixed velocity file(s).

* **OUTPUTFOLDER**: Defines the location of the output folder.

* **NUMBER_OF_CPU_THREADS** *numCpuThreads*: Defines the number of CPU threads for the solver.
* SIMULATION_ON_GPU *1/0*: Defines if the simulation runs on a GPU device.
* GPU_DEVICE *deviceID*: Defines the ID of the GPU device.
* GPU_THREADS_PER_BLOCK *numThreadsPerBlock*: Defines the number of GPU threads per GPU blocks.
* GPU_NUMBER_OF_BLOCKS *numBlock*: Defines the number of GPU blocks.

* ADJUST_ANGLE_X *radAngle*: Defines the angle to adjust anisotropy axis around axis X in radians.
* ADJUST_ANGLE_Y *radAngle*: Defines the angle to adjust anisotropy axis around axis Y in radians.
* ADJUST_ANGLE_Z *radAngle*: Defines the angle to adjust anisotropy axis around axis Z in radians.

* **START_TIME** *startTime*: Defines the starting time of the simulation in seconds.
* **END_TIME** *endTime*: Defines the end time of the simulation in seconds.
* **TIMESTEP** *timeStep*: Defines the time step for the solver in seconds.
* GLOBAL_DAMPING *globalDamping*: Defines the global damping for the solver. It ranges between 0.0 and 1.0.

* **SAVE_INTERVAL** *saveInterval*: Defines the save interval for the solver in seconds.

* SAVE_NODES *1/0*: Inidicates to save a result file on the nodes. The saved data is defined by the following keywords:
  * SAVE_NODE_ID *1/0*: Indicates to save the node ID.
  * SAVE_NODE_COMPONENT *1/0*: Indicates to save the component ID to which the node belongs.
  * SAVE_NODE_MASS *1/0*: Indicates to save the nodal mass.
  * SAVE_NODE_FORCE *1/0*: Indicates to save the nodal force.
  * SAVE_NODE_VELOCITY *1/0*: Indicates to save the nodal velocity.
  * SAVE_NODE_ACCELERATION *1/0*: Indicates to save the nodal acceleration.
  * SAVE_NODE_FIXED_VELOCITY *1/0*: Indicates to save the fixed velocity applied onto the node.
  * SAVE_NODE_BOUNDARIES *1/0*: Indicates to save the applied boundary conditions onto the node.

* SAVE_CELLS *1/0*: Indicates to save a result file on the cells. The saved data is defined by the following keywords:
  * SAVE_CELL_ID *1/0*: Indicates to save the cell ID.
  * SAVE_CELL_COMPONENT *1/0*: Indicates to save the component to which the cell belongs.
  * SAVE_CELL_STATUS *1/0*: Indicates to save the status of the cell (1 - active; 2 - damaged).
  * SAVE_CELL_MATERIAL_PROPERTY *1/0*: Indicates to save the material property index.
  * SAVE_CELL_VOLUME *1/0*: Indicates to save the cell volume.
  * SAVE_CELL_MASS *1/0*: Indicates to save the cell mass.
  * SAVE_CELL_NODE_ID *1/0*: Indicates to save the node IDs that belong to the cell.
  * SAVE_CELL_NODE_MASS *1/0*: Indicates to save the nodal masses that belong to the cell.
  * SAVE_CELL_NODE_FORCE *1/0*: Indicates to save the nodal forces that belong to the cell.
  * SAVE_CELL_NODE_VELOCITY *1/0*: Indicates to save the nodal velocities that belong to the cell.
  * SAVE_CELL_NODE_ACCELERATION *1/0*: Indicates to save the nodal velocities that belong to the cell.

* SAVE_FACES *1/0*: Indicates to save a result file on the faces. The saved data is defined by the following keywords:
  * SAVE_FACE_ID *1/0*: Indicates to save the face ID.
  * SAVE_FACE_COMPONENT *1/0*: Indicates to save component to which the face belongs.
  * SAVE_FACE_CELL_ID *1/0*: Indicates to save the cell ID to which the face belings.
  * SAVE_FACE_AREA *1/0*: Indicates to save the area of the face.
  * SAVE_FACE_NORMAL *1/0*: Indicates to save the face normal.

* SAVE_AXIAL_SPRINGS *1/0*: Indicates to save a result file on the axial springs. The saved data is defined by the following keywords:
  * SAVE_AXIAL_SPRING_ID *1/0*: Indicates to save the axial spring ID.
  * SAVE_AXIAL_SPRING_COMPONENT *1/0*: Indicates to save the component to which the axial spring belongs.
  * SAVE_AXIAL_SPRING_CELL_ID *1/0*: Indicates to save the cell ID to which the axial spring belongs to.
  * SAVE_AXIAL_SPRING_TENSILE_STIFFNESS *1/0*: Indicates to save the tensile stiffness of the axial spring.
  * SAVE_AXIAL_SPRING_COMPRESSIVE_STIFFNESS *1/0*: Indicates to save the compressive stiffness of the axial spring.
  * SAVE_AXIAL_SPRING_TENSILE_DAMPING *1/0*: Indicates to save the tensile damping of the axial spring.
  * SAVE_AXIAL_SPRING_COMPRESSIVE_DAMPING *1/0*: Indicates to save the compressive damping of the axial spring.
  * SAVE_AXIAL_SPRING_TENSILE_STRENGTH *1/0*: Indicates to save the tensile strength of the axial spring.
  * SAVE_AXIAL_SPRING_COMPRESSIVE_STRENGTH *1/0*: Indicates to save the compressive strength of the axial spring.
  * SAVE_AXIAL_SPRING_LOADCASE *1/0*: Indicates to save the load case of the axial spring (0 - not loaded; 1 - tension; 2 - compression).
  * SAVE_AXIAL_SPRING_LENGTH *1/0*: Indicates to save the length of the axial spring.
  * SAVE_AXIAL_SPRING_SPRING_FORCE *1/0*: Indicates to save the spring force of the axial spring.
  * SAVE_AXIAL_SPRING_DAMPING_FORCE *1/0*: Indicates to save the damping force of the axial spring.
  * SAVE_AXIAL_SPRING_TOTAL_FORCE *1/0*: Indicates to save the total (spring + damping) force of the axial spring.

* SAVE_ROTATIONAL_SPRINGS *1/0*: Indicates to save a result file on the rotational springs. The saved data is defined by the following keywords:
  * SAVE_ROTATIONAL_SPRING_ID *1/0*: Indicates to save the rotational spring ID.
  * SAVE_ROTATIONAL_COMPONENT *1/0*: Indicates to save the component to which the rotational spring belongs.
  * SAVE_ROTATIONAL_SPRING_CELL_ID *1/0*: Indicates to save the cell ID to which the rotational spring belongs.
  * SAVE_ROTATIONAL_SPRING_STIFFNESS *1/0*: Indicates to save the stiffness of the rotational spring.
  * SAVE_ROTATIONAL_SPRING_ANGLE *1/0*: Indicates to save the angle of the rotational spring.
  * SAVE_ROTATIONAL_SPRING_FORCE *1/0*: Indicates to save the spring force of the rotational spring.

* SAVE_CONTACTS *1/0*: Indicates to save a result file on the contacts. The save data is defined by the following keywords:
  * SAVE_CONTACT_ID *1/0*: Indicates to save the contact ID.
  * SAVE_CONTACT_FACE *1/0*: Indicates to save the face ID involved in the interaction.
  * SAVE_CONTACT_FRICTION *1/0*: Indicates to save the coefficient of static friction assigned to the interaction.
  * SAVE_CONTACT_NORMAL_STIFFNESS *1/0*: Indicates to save the normal stiffness assigned to the interaction.
  * SAVE_CONTACT_TANGENTIAL_STIFFNESS *1/0*: Indicates to save the tangential stiffness assigned to the interaction.
  * SAVE_CONTACT_NORMAL_OVERLAP *1/0*: Indicates to save the normal overlap of the interaction.
  * SAVE_CONTACT_TANGENTIAL_OVERLAP *1/0*: Indicates to save the tangential overlap of the interaction.
  * SAVE_CONTACT_NORMAL_FORCE *1/0*: Indicates to save the normal force of the interaction.
  * SAVE_CONTACT_TANGENTIAL_FORCE *1/0*: Indicates to save the tangential force of the interaction.
  * SAVE_CONTACT_TOTAL_FORCE *1/0*: Indicates to save the total (normal + tangential) force of the interaction.

* SAVE_EXTERNAL_FORCES *1/0*: Indicates to save a result file on the external forces. The saved data is defined by the following keywords:
  * SAVE_EXTERNAL_FORCE_ID *1/0*: Indicates to save the external force ID.
  * SAVE_EXTERNAL_FORCE_TYPE *1/0*: Indicates to save the type of the external force (0 - gravitational; 1 - global; 2 - component; 3 - plane; 4 - local).
  * SAVE_EXTERNAL_FORCE_VALUE *1/0*: Indicates to save the value of the external force.

An exmple Settings.txt file:

```
NUMBER_OF_COMPONENTS 1

CELLS 1
0_INPUT/Cube_Size_0_01.vtk

MATERIALS 1
0_INPUT/Cube_Materials.txt

INTERACTIONS 0

INITIAL_CONDITION 0

BOUNDARY_CONDITION 1
0_INPUT/Cube_BoundaryCondition.txt

EXTERNAL_FORCE 1
0_INPUT/Cube_ExternalForce.txt

FIXED_VELOCITY 0

OUTPUTFOLDER
1_OUTPUT/SingleCubeTensileTest/

NUMBER_OF_CPU_THREADS 12

SIMULATION_ON_GPU 0
GPU_DEVICE 0
GPU_THREADS_PER_BLOCK 256
GPU_NUMBER_OF_BLOCKS 128

ADJUST_ANGLE_X 0.0174533
ADJUST_ANGLE_Y 0.0174533
ADJUST_ANGLE_Z 0.0174533

START_TIME 0.0
END_TIME 1e-1
TIMESTEP 1e-5
GLOBAL_DAMPING 0.0

SAVE_INTERVAL 1e-3

SAVE_NODES 1
SAVE_NODE_ID 1
SAVE_NODE_COMPONENT 1
SAVE_NODE_MASS 1
SAVE_NODE_FORCE 1
SAVE_NODE_VELOCITY 1
SAVE_NODE_ACCELERATION 1
SAVE_NODE_FIXED_VELOCITY 1
SAVE_NODE_BOUNDARIES 1

SAVE_CELLS 1
SAVE_CELL_ID 1
SAVE_CELL_COMPONENT 1
SAVE_CELL_STATUS 1
SAVE_CELL_MATERIAL_PROPERTY 1
SAVE_CELL_VOLUME 1
SAVE_CELL_MASS 1
SAVE_CELL_NODE_ID 1
SAVE_CELL_NODE_MASS 1
SAVE_CELL_NODE_FORCE 1
SAVE_CELL_NODE_VELOCITY 1
SAVE_CELL_NODE_ACCELERATION 1

SAVE_FACES 1
SAVE_FACE_ID 1
SAVE_FACE_COMPONENT 1
SAVE_FACE_CELL_ID 1
SAVE_FACE_AREA 1
SAVE_FACE_NORMAL 1

SAVE_AXIAL_SPRINGS 1
SAVE_AXIAL_SPRING_ID 1
SAVE_AXIAL_SPRING_COMPONENT 1
SAVE_AXIAL_SPRING_CELL_ID 1
SAVE_AXIAL_SPRING_TENSILE_STIFFNESS 1
SAVE_AXIAL_SPRING_COMPRESSIVE_STIFFNESS 1
SAVE_AXIAL_SPRING_TENSILE_DAMPING 1
SAVE_AXIAL_SPRING_COMPRESSIVE_DAMPING 1
SAVE_AXIAL_SPRING_TENSILE_STRENGTH 1
SAVE_AXIAL_SPRING_COMPRESSIVE_STRENGTH 1
SAVE_AXIAL_SPRING_LOADCASE 1
SAVE_AXIAL_SPRING_LENGTH 1
SAVE_AXIAL_SPRING_SPRING_FORCE 1
SAVE_AXIAL_SPRING_DAMPING_FORCE 1
SAVE_AXIAL_SPRING_TOTAL_FORCE 1

SAVE_ROTATIONAL_SPRINGS 1
SAVE_ROTATIONAL_SPRING_ID 1
SAVE_ROTATIONAL_COMPONENT 1
SAVE_ROTATIONAL_SPRING_CELL_ID 1
SAVE_ROTATIONAL_SPRING_STIFFNESS 1
SAVE_ROTATIONAL_SPRING_ANGLE 1
SAVE_ROTATIONAL_SPRING_FORCE 1

SAVE_CONTACTS 0
SAVE_CONTACT_ID 0
SAVE_CONTACT_FACE 0
SAVE_CONTACT_FRICTION 0
SAVE_CONTACT_NORMAL_STIFFNESS 0
SAVE_CONTACT_TANGENTIAL_STIFFNESS 0
SAVE_CONTACT_NORMAL_OVERLAP 0
SAVE_CONTACT_TANGENTIAL_OVERLAP 0
SAVE_CONTACT_NORMAL_FORCE 0
SAVE_CONTACT_TANGENTIAL_FORCE 0
SAVE_CONTACT_TOTAL_FORCE 0

SAVE_EXTERNAL_FORCES 1
SAVE_EXTERNAL_FORCE_ID 1
SAVE_EXTERNAL_FORCE_TYPE 1
SAVE_EXTERNAL_FORCE_VALUE 1
```

#### Cells input file structure



#### Materials input file structure


```
NUMBER
1

LOCATION
0.0 0.0 0.0

DENSITY
1200

ANISOTROPY_AXIS
1.0 0.0 0.0 
0.0 1.0 0.0
0.0 0.0 1.0

ANISOTROPY_TENSILE_STIFFNESS
1e0
1e0
1e0

ANISOTROPY_COMPRESSIVE_STIFFNESS
1e0
1e0
1e0

ANISOTROPY_TENSILE_DAMPING
1e-2
1e-2
1e-2

ANISOTROPY_COMPRESSIVE_DAMPING
1e-2
1e-2
1e-2

ANISOTROPY_ROT_STIFFNESS
1e0
1e0
1e0

ANISOTROPY_SPRING_TENSILE_STRENGTH
1e3
1e3
1e3

ANISOTROPY_SPRING_COMPRESSIVE_STRENGTH
1e3
1e3
1e3
```


#### Interactions input file structure

#### Initial conditions input file structure

#### Boundary conditions input file structure

#### External forces input file structure

#### Fixed velocities input file structure


### Solver

### Result files

## Small examples

## Theory & background

## Performance tests

## Corn stalk simulation

## Limitations

## Future work & development

## References

[1. Oussama Jarrousse: Modified Mass-Spring System for Physically Based Deformation Modeling, Karlsruhe Transactions on Biomedical Engineering, Vol 14, 2011](https://www.researchgate.net/publication/342899408_Modified_Mass-spring_System_for_Physically_Based_Deformation_Modeling)



