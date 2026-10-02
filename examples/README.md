# FELLOWSHIP OF THE RING (User Example)

*ANATOMÍA DE FRODO*

FRODO tiene varias subclases que le ayudan a cargar con el peso de su destino:
- READERS: Esta subclase contiene una clase necesaria para leer cada formato de anillo que FRODO puede tolerar. Dicha clase se instancia en db.reader cuando se especifica inicialmente el formato. A fecha de 22/09/2026, FRODO tolera los siguientes formatos de anillos:
    - CODA
    - CODA_SINGLE (una malla distinta por caso; es el formato de los estudios GCI)
    - NUMPY (archivos .npy, requiere `file=`)
    - NUMPYFILE (obsoleto, requiere `file=`)
    - PYLOM (requiere `file=`)
    - HORSES3D (acepta `strict=`)

- RESIDUALS: Es la parte instrospectiva de nuestro protagonista. Como muchos sabréis, es una habilidad cada vez menos común en la sociedad hoy en día y difícil de adquirir por la abundancia de mediocridad. Esta escasez se traslada a los formatos de anillos que tenemos. Actualmente (22/09/2026) sólo CODA, CODA_SINGLE (que reutiliza la misma clase que CODA) y HORSES3D permiten leer y analizar el proceso de cálculo de las simulaciones.
- SETS: Venga, que llegamos a la parte heróica. Llega un momento que, por muy jugoso que sea el anillo (los datos), hay que soltarlo. Esa fuerza interior reside en esta subclase. En ella se nos permite arrojar nuestro granito de arena añadiendo cálculos indirectos al dataset y exportarlos a los formatos disponibles. De nuevo, también está la opción de tirar a Smeagol con el anillo encima (exportar a formato pyLOM), y como en la obra original, esto también se ha dominado para que salga bien. Lo tienen todos los formatos menos HORSES3D.
- STATS: La cuarta pieza, y la última en llegar. Cuando ya tienes el anillo en la mano, toca preguntarse si los números se sostienen: estadísticos entre casos, diferencias entre stages y, en CODA_SINGLE, el estudio de convergencia de malla (Richardson y GCI). No lo tienen ni PYLOM, ni NUMPYFILE, ni HORSES3D.

Ojo con una regla que no se ve en el código: **lo que llames como `db.algo(...)` casi nunca vive en FRODO**. Se busca en `sets`, `reader`, `residuals` y `stats`, en ese orden, y gana el primero que lo tenga. Por eso la misma llamada puede tener firma distinta según el formato con el que abriste la base.

*ANATOMÍA DE SAM*

Como ya hemos dicho, SAM es ese compañero indispensable de viaje. El mítico compañero de trabajo que habla poco, pero te presenta una solución al apartado que no sabíais hacer ninguno. El callado de clase que misteriosamente saca un 5 en el examen pero que os ha explicado lo más importante de la asignatura para que aprovéis. El héroe sin capa, tan capaz de sostener las emociones del grupo como de enfrentarse a lo orcos más feos. Sus capacidades (clases) se dividen en:
- Gardener: SAM es capaz de organizar todos los datos de FRODO para crear diccionarios de tensores que transformaremos en Datasets. Cocina los datos por nosotros para ponerlos, vengan de donde vengan. También puede aligerarlos (reducir por frecuencia).

- HDF5reader: Un lector básico de archivos .h5 que facilita su estudio.

- Backpack: La mochila mítica del personaje donde encuentras cosas que es mejor llevar y no necesitar, que necesitar y no llevar (como los condones). Aquí encontraremos métodos estáticos que FRODO usará de vez en cuando sin pisparse. Algunos son muy generales y otros específicos, pero pocas veces los tendremos que usar nosotros como usuarios.

- Weapons: La parte más matemática (guerrera) del personaje. Métodos para ordenar geometrías 2D y 3D, métodos de derivación numérica y gradientes y un algoritmo para estudios con GMM. Lo que te daba pereza estudiar, pero que lo necesitas para todo. Aquí viven también los núcleos de interpolación malla-a-malla que usan los SETS por debajo.

- DifferentialOperators: El hermano callado de Weapons. Gradiente, jacobiano y divergencia sobre nubes de puntos dispersas por mínimos cuadrados móviles — es decir, **sin necesitar conectividad ni que los puntos estén ordenados**. Cuando la malla no está o no te fías de ella, es lo que quieres.

- DictVisualizer: La parte más ilustrativa (emocional) del personaje. Como FRODO trabaja mucho con diccionarios, aquí encontramos tres métodos gráficos para resumir la información de estos y que no nos comamos la cabeza con los dramas de FRODO.

Un detalle práctico: SAM **no se instancia nunca**. Todo se llama como `SAM.Area.funcion(...)`, y la única excepción es `SAM.HDF5reader`, que sí es un objeto porque guarda el fichero abierto.

En adelante, y como ahora no dispongo de más tiempo, resumiré la ayuda mínima necesaria en comentarios mientras navegas por este notebook. No dudes en escribirme si tienes dudas sobre la convivencia en esta comunidad (miguel.jaraizga@upm.es)