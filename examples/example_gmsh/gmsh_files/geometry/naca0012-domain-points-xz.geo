// DOMAIN points
// From Point(1) to Point(9)

//--------------------------------------------------//
// Some special dimensions
//--------------------------------------------------//
x0 = 0.0;            // Domain center: x-coordinate
y0 = 0.0;            // Domain center: y-coordinate
z0 = 0.0;            // Domain center: z-coordinate
R0 = 100.0;          // Disk radius
L0 = 10.0;           // Length of inner square domain
h0 = 1.0;            // Height of disk
k0 = 0.5*Sqrt(2.0);

lx = L0-x0;          // Square domain: left x-coordinate
lz = L0-z0;          // Square domain: left z-coordinate
rx = L0+x0;          // Square domain: right x-coordinate
rz = L0+z0;          // Square domain: right z-coordinate
cx = k0*R0;          // Disk domain:   x-coordinate
cz = k0*R0;          // Disk domain:   z-coordinate

//--------------------------------------------------//
// Points: Outer Boundary
//--------------------------------------------------//
Point(1) = {+x0,   y0,+z0,   lc};
Point(2) = {-lx,   y0,-lz,   lc};
Point(3) = {+rx,   y0,-rz,   lc};
Point(4) = {+rx,   y0,+rz,   lc};
Point(5) = {-lx,   y0,+lz,   lc};
Point(6) = {+cx+x0,y0,+cz+z0,lc};
Point(7) = {-cx+x0,y0,+cz+z0,lc};
Point(8) = {-cx+x0,y0,-cz+z0,lc};
Point(9) = {+cx+x0,y0,-cz+z0,lc};
