// Gmsh project created on Wed Dec 08 20:58:56 2021
SetFactory("OpenCASCADE");
//+
Ellipse(1) = {0.0, 0.0, 0.0, 0.00975, 0.0082, 0, 2*Pi};
//+
Extrude {0, 0, 0.005} {
  Curve{1}; 
}
//+
Curve Loop(2) = {3};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3};
//+
Curve Loop(4) = {1};
//+
Plane Surface(3) = {4};
//+
Surface Loop(1) = {2, 1, 3};
//+
Volume(1) = {1};
