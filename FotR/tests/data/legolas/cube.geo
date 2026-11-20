Point(1)={0,0,0,1};Point(2)={1,0,0,1};Point(3)={1,1,0,1};Point(4)={0,1,0,1};
Line(1)={1,2};Line(2)={2,3};Line(3)={3,4};Line(4)={4,1};
Curve Loop(1)={1,2,3,4};Plane Surface(1)={1};
Transfinite Curve{1,2,3,4}=2;Transfinite Surface{1};Recombine Surface{1};
out[]=Extrude{0,0,1}{Surface{1};Layers{1};Recombine;};
Physical Volume("fluid")={out[1]};
Physical Surface("bottom")={1};Physical Surface("top")={out[0]};
Physical Surface("sides")={out[2],out[3],out[4],out[5]};
