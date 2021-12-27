#################################################################
# Project: MicroCrop - Advanced Anisotropic Mass-Spring System
# Author: Adam Kovacs
# Version: 1.0.0
# Maintainer: Adam Kovacs
# E-mail: kovadam19@gmail.com
# Released: 01 January 2022
#################################################################

# Imports
import numpy as np

# Node class
class Node:
    def __init__(self, x, y, z):
        self.position = np.array([x, y, z])

# Cell class
class Cell:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.center = np.array([0.0, 0.0, 0.0])

        self.volume = 0.0
        self.density = 0.0
        self.mass = 0.0
        self.axes = []
        self.axial_tensile_stiffnesses = []
        self.axial_compressive_stiffnesses = []
        self.axial_tensile_dampings = []
        self.axial_compressive_dampings = []
        self.rotational_stiffnesses = []
        self.tensile_strengths = []
        self.compressive_strengths = []
        self.time_step = 1e15

# Create materials class
class CreateMaterials:
    def __init__(self):
        # Input mesh
        self.mesh_file = "CornStalk_Mesh.vtk"
        
        # Output files
        self.vtk_file = "CornStalk_Materials.vtk"
        self.text_file = "CornStalk_Materials.txt"
        
        # Center point and orientation of the corn stalk section
        self.stalk_center_point = np.array([0.0, 0.0, 0.0])
        self.longitudinal_axis = 2
        
        # Critical damping ratio
        self.critical_damping_ratio = 0.1
        
        # Major and minor radii of the core of the corn stalk section
        self.core_major_radius = 8.75e-3
        self.core_minor_radius = 7.2e-3

        # Densities
        self.core_density = 202.1
        self.skin_density = 1501.5

        # Core stiffnesses
        self.core_tensile_stiffness_radial = 1.32e5
        self.core_tensile_stiffness_tangential = 1.32e5
        self.core_tensile_stiffness_longitudinal = 4.39e5

        self.core_compressive_stiffness_radial = 1.32e5
        self.core_compressive_stiffness_tangential = 1.32e5
        self.core_compressive_stiffness_longitudinal = 4.39e5
        
        # Core rotational stiffness
        self.core_rotational_stiffness = 10.0

        # Core strengths
        self.core_tensile_strength_radial = 1000
        self.core_tensile_strength_tangential = 4.5
        self.core_tensile_strength_longitudinal = 1000

        self.core_compressive_strength_radial = 1000
        self.core_compressive_strength_tangential = 9.0
        self.core_compressive_strength_longitudinal = 1000

        # Skin stiffnesses
        self.skin_tensile_stiffness_radial = 1.32e6
        self.skin_tensile_stiffness_tangential = 1.32e6
        self.skin_tensile_stiffness_longitudinal = 4.39e6

        self.skin_compressive_stiffness_radial = 1.32e6
        self.skin_compressive_stiffness_tangential = 1.32e6
        self.skin_compressive_stiffness_longitudinal = 4.39e6
        
        # Sking rotational stiffnes
        self.skin_rotational_stiffness = 10.0

        # Skin strengths
        self.skin_tensile_strength_radial = 1000
        self.skin_tensile_strength_tangential = 15.0
        self.skin_tensile_strength_longitudinal = 1000

        self.skin_compressive_strength_radial = 1000
        self.skin_compressive_strength_tangential = 30.0
        self.skin_compressive_strength_longitudinal = 1000
        
        # Container for the nodes and cells
        self.nodes = []
        self.cells = []
    
    # Reading the corn stalk mesh
    def read_mesh(self):
        with open(self.mesh_file) as file:
            line = file.readline()
            while line:
                line = file.readline()
                if "POINTS" in line:
                    item, number, data_type = line.split()
                    number = int(number)
                    for i in range(number):
                        x, y, z = file.readline().split()
                        x = float(x)
                        y = float(y)
                        z = float(z)
                        new_node = Node(x, y, z)
                        self.nodes.append(new_node)

                if "CELLS" in line:
                    item, number, data_number = line.split()
                    number = int(number)
                    for i in range(number):
                        data_line = file.readline()
                        if data_line.startswith("4"):
                            cell_type, a, b, c, d = data_line.split()
                            a = int(a)
                            b = int(b)
                            c = int(c)
                            d = int(d)
                            new_cell = Cell(a, b, c, d)
                            self.cells.append(new_cell)
    
    # Calculating the barycenter of the cells
    def calculate_cell_center(self):
        for cell in self.cells:
            node_a_position = self.nodes[cell.a].position
            node_b_position = self.nodes[cell.b].position
            node_c_position = self.nodes[cell.c].position
            node_d_position = self.nodes[cell.d].position
            location = node_a_position + node_b_position + node_c_position + node_d_position
            cell.center = 0.25 * location
    
    # Calculating the cell volume
    def calculate_cell_volume(self):
        for cell in self.cells:
            node_a_position = self.nodes[cell.a].position
            node_b_position = self.nodes[cell.b].position
            node_c_position = self.nodes[cell.c].position
            node_d_position = self.nodes[cell.d].position

            vector_ab = node_b_position - node_a_position
            vector_ac = node_c_position - node_a_position
            vector_ad = node_d_position - node_a_position

            cross_product_ab_ac = np.cross(vector_ab, vector_ac)

            volume_parallelepipedon = np.dot(vector_ad, cross_product_ab_ac)

            cell.volume = np.abs(volume_parallelepipedon / 6.0)
    
    # Assigning cell material
    def assign_cell_material(self):
        for cell in self.cells:
            # Calculating the X and Y coordinates of the cell center
            stalk_center = self.stalk_center_point
            stalk_center[self.longitudinal_axis] = cell.center[self.longitudinal_axis]
            x = ((cell.center[0] - stalk_center[0]) * (cell.center[0] - stalk_center[0])) / (self.core_major_radius * self.core_major_radius)
            y = ((cell.center[1] - stalk_center[1]) * (cell.center[1] - stalk_center[1])) / (self.core_minor_radius * self.core_minor_radius)
            
            # Check if the cell center is in the core
            if x + y <= 1.0:
                # CORE #
                # Assign the density and calculate the cell mass
                cell.density = self.core_density
                cell.mass = cell.density * cell.volume
                
                # Calculating the radial, tangential and longitudinal axes
                radial_axis = (cell.center - stalk_center) / np.linalg.norm(cell.center - stalk_center)
                tangential_axis = np.array([-radial_axis[1], radial_axis[0], radial_axis[2]])
                longitudinal_axis = np.cross(radial_axis, tangential_axis)
                cell.axes.append(radial_axis)
                cell.axes.append(tangential_axis)
                cell.axes.append(longitudinal_axis)
                
                # Assigning the tensile stiffnesses
                cell.axial_tensile_stiffnesses.append(self.core_tensile_stiffness_radial)
                cell.axial_tensile_stiffnesses.append(self.core_tensile_stiffness_tangential)
                cell.axial_tensile_stiffnesses.append(self.core_tensile_stiffness_longitudinal)
                
                # Calculating the tensile damping ratios
                cell.axial_tensile_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.core_tensile_stiffness_radial * cell.mass))
                cell.axial_tensile_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.core_tensile_stiffness_tangential * cell.mass))
                cell.axial_tensile_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.core_tensile_stiffness_longitudinal * cell.mass))
                
                # Assigning the compressive stiffnesses
                cell.axial_compressive_stiffnesses.append(self.core_compressive_stiffness_radial)
                cell.axial_compressive_stiffnesses.append(self.core_compressive_stiffness_tangential)
                cell.axial_compressive_stiffnesses.append(self.core_compressive_stiffness_longitudinal)
                
                # Calculating the compressive damping ratios
                cell.axial_compressive_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.core_compressive_stiffness_radial * cell.mass))
                cell.axial_compressive_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.core_compressive_stiffness_tangential * cell.mass))
                cell.axial_compressive_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.core_compressive_stiffness_longitudinal * cell.mass))
                
                # Assigning the rotational stiffnesses
                cell.rotational_stiffnesses.append(self.core_rotational_stiffness)
                cell.rotational_stiffnesses.append(self.core_rotational_stiffness)
                cell.rotational_stiffnesses.append(self.core_rotational_stiffness)
                
                # Assigning the tensile strengths
                cell.tensile_strengths.append(self.core_tensile_strength_radial)
                cell.tensile_strengths.append(self.core_tensile_strength_tangential)
                cell.tensile_strengths.append(self.core_tensile_strength_longitudinal)
                
                # Assigning the compressive strengths
                cell.compressive_strengths.append(self.core_compressive_strength_radial)
                cell.compressive_strengths.append(self.core_compressive_strength_tangential)
                cell.compressive_strengths.append(self.core_compressive_strength_longitudinal)
                
                # Calculating the critical time step for each cell
                for stiffness in cell.axial_tensile_stiffnesses:
                    time_step = np.sqrt(cell.mass / stiffness)
                    if time_step < cell.time_step:
                        cell.time_step = time_step

                for stiffness in cell.axial_compressive_stiffnesses:
                    time_step = np.sqrt(cell.mass / stiffness)
                    if time_step < cell.time_step:
                        cell.time_step = time_step

            else:
                # SKIN #
                # Assign the density and calculate the cell mass
                cell.density = self.skin_density
                cell.mass = cell.density * cell.volume
                
                # Calculating the radial, tangential and longitudinal axes
                radial_axis = (cell.center - stalk_center) / np.linalg.norm(cell.center - stalk_center)
                tangential_axis = np.array([-radial_axis[1], radial_axis[0], radial_axis[2]])
                longitudinal_axis = np.cross(radial_axis, tangential_axis)
                cell.axes.append(radial_axis)
                cell.axes.append(tangential_axis)
                cell.axes.append(longitudinal_axis)
                
                # Assigning the tensile stiffnesses
                cell.axial_tensile_stiffnesses.append(self.skin_tensile_stiffness_radial)
                cell.axial_tensile_stiffnesses.append(self.skin_tensile_stiffness_tangential)
                cell.axial_tensile_stiffnesses.append(self.skin_tensile_stiffness_longitudinal)
                
                # Calculating the tensile damping ratios
                cell.axial_tensile_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.skin_tensile_stiffness_radial * cell.mass))
                cell.axial_tensile_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.skin_tensile_stiffness_tangential * cell.mass))
                cell.axial_tensile_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.skin_tensile_stiffness_longitudinal * cell.mass))
                
                # Assigning the compressive stiffnesses
                cell.axial_compressive_stiffnesses.append(self.skin_compressive_stiffness_radial)
                cell.axial_compressive_stiffnesses.append(self.skin_compressive_stiffness_tangential)
                cell.axial_compressive_stiffnesses.append(self.skin_compressive_stiffness_longitudinal)
                
                # Calculating the compressive damping ratios
                cell.axial_compressive_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.skin_compressive_stiffness_radial * cell.mass))
                cell.axial_compressive_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.skin_compressive_stiffness_tangential * cell.mass))
                cell.axial_compressive_dampings.append(
                    self.critical_damping_ratio * 2.0 * np.sqrt(self.skin_compressive_stiffness_longitudinal * cell.mass))

                # Assigning the rotational stiffnesses
                cell.rotational_stiffnesses.append(self.skin_rotational_stiffness)
                cell.rotational_stiffnesses.append(self.skin_rotational_stiffness)
                cell.rotational_stiffnesses.append(self.skin_rotational_stiffness)

                # Assigning the tensile strengths
                cell.tensile_strengths.append(self.skin_tensile_strength_radial)
                cell.tensile_strengths.append(self.skin_tensile_strength_tangential)
                cell.tensile_strengths.append(self.skin_tensile_strength_longitudinal)

                # Assigning the compressive strengths
                cell.compressive_strengths.append(self.skin_compressive_strength_radial)
                cell.compressive_strengths.append(self.skin_compressive_strength_tangential)
                cell.compressive_strengths.append(self.skin_compressive_strength_longitudinal)

                # Calculating the critical time step for each cell
                for stiffness in cell.axial_tensile_stiffnesses:
                    time_step = np.sqrt(cell.mass / stiffness)
                    if time_step < cell.time_step:
                        cell.time_step = time_step

                for stiffness in cell.axial_compressive_stiffnesses:
                    time_step = np.sqrt(cell.mass / stiffness)
                    if time_step < cell.time_step:
                        cell.time_step = time_step
    
    # Writing the materials into VTK file
    def write_VTK_file(self):
        with open(self.vtk_file, "w") as file:
            file.write("# vtk DataFile Version 4.2\n")
            file.write("vtk output\n")
            file.write("ASCII\n")
            file.write("DATASET UNSTRUCTURED_GRID\n")
            file.write("\n")

            total_number_of_points = len(self.cells)

            file.write("POINTS " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.center[0]) + " " + str(cell.center[1]) + " " + str(cell.center[2]) + "\n")
            file.write("\n")

            file.write("CELLS " + str(total_number_of_points) + " " + str(total_number_of_points * 2) + "\n")
            for i in range(total_number_of_points):
                file.write("1" + " " + str(i) + "\n")
            file.write("\n")

            file.write("CELL_TYPES " + str(total_number_of_points) + "\n")
            for i in range(total_number_of_points):
                file.write("1\n")
            file.write("\n")

            file.write("POINT_DATA " + str(total_number_of_points) + "\n")
            file.write("FIELD FieldData " + str(26) + "\n")
            file.write("\n")

            file.write("Volume_(m3) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.volume) + "\n")
            file.write("\n")

            file.write("Density_(kg/m3) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.density) + "\n")
            file.write("\n")

            file.write("Mass_(kg) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.mass) + "\n")
            file.write("\n")

            file.write("Timestep_(sec) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.time_step) + "\n")
            file.write("\n")

            file.write("Axis_0_(-) " + str(3) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axes[0][0]) + " " + str(cell.axes[0][1]) + " " + str(cell.axes[0][2]) + "\n")
            file.write("\n")

            file.write("Axis_1_(-) " + str(3) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axes[1][0]) + " " + str(cell.axes[1][1]) + " " + str(cell.axes[1][2]) + "\n")
            file.write("\n")

            file.write("Axis_2_(-) " + str(3) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axes[2][0]) + " " + str(cell.axes[2][1]) + " " + str(cell.axes[2][2]) + "\n")
            file.write("\n")

            file.write("TensileStiffness_0_(N/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_tensile_stiffnesses[0]) + "\n")
            file.write("\n")

            file.write("TensileStiffness_1_(N/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_tensile_stiffnesses[1]) + "\n")
            file.write("\n")

            file.write("TensileStiffness_2_(N/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_tensile_stiffnesses[2]) + "\n")
            file.write("\n")

            file.write("CompressiveStiffness_0_(N/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_compressive_stiffnesses[0]) + "\n")
            file.write("\n")

            file.write("CompressiveStiffness_1_(N/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_compressive_stiffnesses[1]) + "\n")
            file.write("\n")

            file.write("CompressiveStiffness_2_(N/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_compressive_stiffnesses[2]) + "\n")
            file.write("\n")

            file.write("TensileDamping_0_(Ns/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_tensile_dampings[0]) + "\n")
            file.write("\n")

            file.write("TensileDamping_1_(Ns/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_tensile_dampings[1]) + "\n")
            file.write("\n")

            file.write("TensileDamping_2_(Ns/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_tensile_dampings[2]) + "\n")
            file.write("\n")

            file.write("CompressiveDamping_0_(Ns/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_compressive_dampings[0]) + "\n")
            file.write("\n")

            file.write("CompressiveDamping_1_(Ns/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_compressive_dampings[1]) + "\n")
            file.write("\n")

            file.write("CompressiveDamping_2_(Ns/m) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.axial_compressive_dampings[2]) + "\n")
            file.write("\n")

            file.write("RotationalStiffness_(N/rad) " + str(3) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.rotational_stiffnesses[0]) + str(cell.rotational_stiffnesses[1]) + str(cell.rotational_stiffnesses[2]) + "\n")
            file.write("\n")

            file.write("Tensile_Strength_0_(N) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.tensile_strengths[0]) + "\n")
            file.write("\n")

            file.write("Tensile_Strength_1_(N) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.tensile_strengths[1]) + "\n")
            file.write("\n")

            file.write("Tensile_Strength_2_(N) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.tensile_strengths[2]) + "\n")
            file.write("\n")

            file.write("Compressive_Strength_0_(N) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.compressive_strengths[0]) + "\n")
            file.write("\n")

            file.write("Compressive_Strength_1_(N) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.compressive_strengths[1]) + "\n")
            file.write("\n")

            file.write("Compressive_Strength_2_(N) " + str(1) + " " + str(total_number_of_points) + " double\n")
            for cell in self.cells:
                file.write(str(cell.compressive_strengths[2]) + "\n")
            file.write("\n")
    
    # Writing the materials into TXT file
    def write_text_file(self):
        with open(self.text_file, "w") as file:
            file.write("NUMBER\n")
            file.write(str(len(self.cells)) + "\n")
            file.write("\n")

            file.write("LOCATION\n")
            for cell in self.cells:
                file.write(str(cell.center[0]) + " " + str(cell.center[1]) + " " + str(cell.center[2]) + "\n")
            file.write("\n")

            file.write("DENSITY\n")
            for cell in self.cells:
                file.write(str(cell.density) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_AXIS\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.axes[i][0]) + " " + str(cell.axes[i][1]) + " " + str(cell.axes[i][2]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_TENSILE_STIFFNESS\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.axial_tensile_stiffnesses[i]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_COMPRESSIVE_STIFFNESS\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.axial_compressive_stiffnesses[i]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_TENSILE_DAMPING\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.axial_tensile_dampings[i]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_COMPRESSIVE_DAMPING\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.axial_compressive_dampings[i]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_ROT_STIFFNESS\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.rotational_stiffnesses[i]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_SPRING_TENSILE_STRENGTH\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.tensile_strengths[i]) + "\n")
            file.write("\n")

            file.write("ANISOTROPY_SPRING_COMPRESSIVE_STRENGTH\n")
            for cell in self.cells:
                for i in range(3):
                    file.write(str(cell.compressive_strengths[i]) + "\n")
            file.write("\n")


# Main
if __name__ == '__main__':
    my_creator = CreateMaterials()
    my_creator.read_mesh()
    my_creator.calculate_cell_center()
    my_creator.calculate_cell_volume()
    my_creator.assign_cell_material()
    my_creator.write_VTK_file()
    my_creator.write_text_file()
    