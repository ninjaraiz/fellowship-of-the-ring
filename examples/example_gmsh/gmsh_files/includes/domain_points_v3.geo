//--------------------------------------------------//
// DOMAIN points: circular farfield only
//--------------------------------------------------//

R0 = c*5.0;          // Disk radius
// h0 = 1.0;            // Height of disk
k0 = 0.5*Sqrt(2.0);  // Scaling factor for circle

If (ExtrudeDirection == 2)

    cx = k0*R0;          
    cz = k0*R0;          

    Point(1) = {x0, y0, z0, lc_ff}; // center (optional)

    // Four points to define the circle
    Point(6) = {+cx+x0, y0, +cz+z0, lc_ff};
    Point(7) = {-cx+x0, y0, +cz+z0, lc_ff};
    Point(8) = {-cx+x0, y0, -cz+z0, lc_ff};
    Point(9) = {+cx+x0, y0, -cz+z0, lc_ff};

ElseIf (ExtrudeDirection == 3)

    cx = k0*R0;          
    cy = k0*R0;          

    Point(1) = {x0, y0, z0, lc_ff}; // center (optional)

    // Four points to define the circle
    Point(6) = {+cx+x0, +cy+y0, z0, lc_ff};
    Point(7) = {-cx+x0, +cy+y0, z0, lc_ff};
    Point(8) = {-cx+x0, -cy+y0, z0, lc_ff};
    Point(9) = {+cx+x0, -cy+y0, z0, lc_ff};

EndIf
