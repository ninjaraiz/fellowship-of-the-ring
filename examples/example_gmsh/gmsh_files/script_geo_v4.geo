//--------------------------------------------------//
// Mesh for simulations of flows around airfoils
//--------------------------------------------------//

SetFactory("OpenCASCADE");

// ---- Extrusión ----
// 2 = extruir en Y | 3 = extruir en Z
ExtrudeDirection = 2;

h0 = 0.1; // altura de extrusión
nLayers = 3; // nº de capas

lc_ff = 1e-1; // 8e-2
lc_a = 5e-4; // 5e-5
lc_ball = 5e-2; // 1e-2 
// lc_box = 8e-3; // 3e-4

use_boundary_layer = 1; // 1 = activar capa límite | 0 = desactivar

//--------------------------------------------------//
// Geometry Tolerance
//--------------------------------------------------//
Geometry.Tolerance = 1.0e-10;

//--------------------------------------------------//
// Number of Threads
//--------------------------------------------------//
General.NumThreads = 4;

//--------------------------------------------------//
// Points: Airfoil-MAIN
//--------------------------------------------------//
Include "./geometry/airfoil_geometry.geo";

//--------------------------------------------------//
// Points: DOMAIN
//--------------------------------------------------//
// - From Point(1) to Point(9)
x0 = 1.0*c ;            // Domain center: x-coordinate
y0 = 0.0;            // Domain center: y-coordinate
z0 = 0.0;            // Domain center: z-coordinate
Include "./includes/domain_points_v3.geo";

//--------------------------------------------------//
// Curves: DOMAIN
//--------------------------------------------------//
Point(200) = {0, 0, 0, lc_box};

// Boundary Curves
Circle(5) = {9,1,8};  // down
Circle(6) = {8,1,7};  // rigth
Circle(7) = {7,1,6};  // up
Circle(8) = {6,1,9};  // left

Curve Loop(1) = {5,6,7,8};

//--------------------------------------------------//
// Curves: Airfoil-MAIN
//--------------------------------------------------//
If (geometry_type == 1)
  Spline(100) = {1001:1001+n_points-2, 1001-2+n_points*2:1001-1+n_points, 1001};
ElseIf (geometry_type == 2)
  Spline(100) = {1001:1000+n_points, 1001};
EndIf

Curve Loop(2) = {100};

//--------------------------------------------------//
// Superficie del dominio = círculo menos airfoil
//--------------------------------------------------//
Plane Surface(1) = {1,2};

//--------------------------------------------------//
// Fields
//--------------------------------------------------//

If (use_boundary_layer)
  Transfinite Curve {100} = 6500 Using Progression 1.0; // 6500 con capa límite. 7000 sin capa límite.

  Field[1] = BoundaryLayer;
  Field[1].CurvesList = {100};   // spline del perfil
  Field[1].Size      = lc_a;
  Field[1].SizeFar   = lc_box;
  Field[1].NbLayers  = 7;
  Field[1].Thickness = 0.05*t;
  Field[1].Ratio     = 1.1;
  Field[1].Quads     = 1;

  // Fans en LE/TE (ajustar IDs según tu geometría)
  Mesh.BoundaryLayerFanElements = 10;
  Field[1].FanPointsList = {1100, 1050};
  // Field[1].FanPointsSizesList = {lc_a/5, lc_a/5};

  BoundaryLayer Field = 1;
EndIf

// 2) Big disk for smooth transition to far-field
Field[2] = Ball;
Field[2].VIn  = lc_ball;
Field[2].VOut = lc_ff;
Field[2].Radius = 2.5*c;
Field[2].Thickness = 3*c;
Field[2].XCenter = 0.0; Field[2].YCenter = 0.0; Field[2].ZCenter = 0.0;

// 5) Wake box (opcional)
lc_box = lc_a * (Field[1].Ratio)^^(Field[1].NbLayers)
Field[7] = Box;
Field[7].VIn  = lc_box;
Field[7].VOut = lc_ff;
Field[7].Thickness = 3*c;
Field[7].XMin = -0.7*c; Field[7].XMax = 0.7*c;
Field[7].YMin = -2.0*c; Field[7].YMax =  2.0*c;
Field[7].ZMin = -t; Field[7].ZMax =  t;

// 6) Combina todo (MIN): toma el tamaño más fino requerido en cada punto,
//    pero como los LE/TE están en Threshold su efecto está espacialmente limitado.

Field[8] = Min;
If (use_boundary_layer) 
  Field[8].FieldsList = {1, 2, 7};
Else
  Field[8].FieldsList = {2, 7};
EndIf

Background Field = 8;

// --- Controles de suavizado y calidad ---
Mesh.Smoothing = 5;             // número de iteraciones de suavizado Laplaciano
// Mesh.MinimumCircleAngle = 30;    // ángulo mínimo en los triángulos (mejora ortogonalidad)
Mesh.CharacteristicLengthExtendFromBoundary = 0;  // transición más suave desde BC

Mesh.CharacteristicLengthFromCurvature = 0.2;

Mesh.CharacteristicLengthFromPoints = 0.006;

// --------------------------------------------------//
// Extrude Mesh
// --------------------------------------------------//
If (ExtrudeDirection == 2) // extrusión en Y
  Extrude {0, h0, 0} {
    Surface{1};
    Layers{nLayers};
    Recombine;
  }
ElseIf (ExtrudeDirection == 3) // extrusión en Z
  Extrude {0, 0, h0} {
    Surface{1};
    Layers{nLayers};
    Recombine;
  }
EndIf

// --------------------------------------------------//
// Physical Entities 3D
// --------------------------------------------------//
Physical Surface("FarField", 1) = {2, 3, 4, 5};
Physical Surface("SymmetryPlane", 2) = {7, 1};
Physical Surface("Airfoil", 3) = {6};
Physical Volume("Domain", 4) = {1};

// Crear mallar y expotar:
// Mesh 3;