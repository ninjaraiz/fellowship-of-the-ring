// DOMAIN points
// From Point(1) to Point(9)

//--------------------------------------------------//
// Some special dimensions
//--------------------------------------------------//
x0 = 0.0;            // Domain center: x-coordinate
y0 = 0.0;            // Domain center: y-coordinate
z0 = 0.0;            // Domain center: z-coordinate
R0 = 500.0;          // Disk radius
L0 = 20.0;           // Length of inner square domain
h0 = 1.0;            // Height of disk
k0 = 0.5*Sqrt(2.0);

lx = L0-x0;          // Square domain: left x-coordinate
ly = L0-y0;          // Square domain: left x-coordinate
rx = L0+x0;          // Square domain: left x-coordinate
ry = L0+y0;          // Square domain: left x-coordinate
cx = k0*R0;          // Disk domain:   x-coordinate
cy = k0*R0;          // Disk domain:   y-coordinate

//--------------------------------------------------//
// Points: Outer Boundary
//--------------------------------------------------//
Point(1) = {+x0,   +y0,   z0,lc};
Point(2) = {-lx,   -ly,   z0,lc};
Point(3) = {+rx,   -ry,   z0,lc};
Point(4) = {+rx,   +ry,   z0,lc};
Point(5) = {-lx,   +ly,   z0,lc};
Point(6) = {+cx+x0,+cy+y0,z0,lc};
Point(7) = {-cx+x0,+cy+y0,z0,lc};
Point(8) = {-cx+x0,-cy+y0,z0,lc};
Point(9) = {+cx+x0,-cy+y0,z0,lc};
