// AIRFOIL GEOMETRY

geometry_type = 2; // 1: "NACA" o 2: "PUNTOS"

If (geometry_type == 1)
    naca_name = 5703;
    c = 1.0; 
    n_points = 200;
    // Extraer los dígitos del perfil
    m = Floor(naca_name / 1000) / 100.0; // Máximo camber
    p = Floor((naca_name % 1000) / 100) / 10.0; // Posición del máximo camber
    t = (naca_name % 100) / 100.0; // Espesor relativo

    // Arrays para almacenar los tags de puntos
    PointTags[] = {};

    // Generar coordenadas del perfil usando ecuaciones NACA
    For i In {0:n_points-1}
        x = c * (i / (n_points-1));
        yt = 5 * t * c * (0.2969 * Sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)^2 + 0.2843 * (x/c)^3 - 0.1015 * (x/c)^4);

        // Calcular coordenadas superior e inferior
        If (x < p * c)
            yc = (m / (p^2)) * (2 * p * (x/c) - (x/c)^2);
            dyc_dx = (2 * m / (p^2)) * (p - x/c);
        Else
            yc = (m / ((1 - p)^2)) * ((1 - 2 * p) + 2 * p * (x/c) - (x/c)^2);
            dyc_dx = (2 * m / ((1 - p)^2)) * (p - x/c);
        EndIf
        theta = Atan(dyc_dx);

        xu = x - yt * Sin(theta);
        yu = yc + yt * Cos(theta);
        xl = x + yt * Sin(theta);
        yl = yc - yt * Cos(theta);

        // Definir puntos de la geometría
        tag_u = 1000 + i;
        tag_l = 1000 + n_points + i;
        If (ExtrudeDirection == 2)
            Point(tag_u) = {xu, y0, yu, lc};
            Point(tag_l) = {xl, y0, yl, lc};
        ElseIf (ExtrudeDirection == 3)
            Point(tag_u) = {xu, yu, z0, lc};
            Point(tag_l) = {xl, yl, z0, lc};
        EndIf
        PointTags += {tag_u, tag_l};
    EndFor

ElseIf (geometry_type == 2)
    Include "points_imported_f22.geo";
EndIf
