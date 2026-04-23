# FELLOWSHIP OF THE RING (User Example)

*ANATOMÍA DE FRODO*

FRODO tiene varias subclases que le ayudan a cargar con el peso de su destino:
- READERS: Esta subclase contiene una clase necesaria para leer cada formato de anillo que FRODO puede tolerar. Dicha clase se instancia en db.reader cuando se especifica inicialmente el formato. A fecha de 19/02/2026, FRODO tolera los siguientes formatos de anillos:
    - CODA
    - NUMPYFILE (archivos .npy)
    - FLUENT (en desarrollo)
    - pyLOM (en desarrollo)

- RESIDUALS: Es la parte instrospectiva de nuestro protagonista. Como muchos sabréis, es una habilidad cada vez menos común en la sociedad hoy en día y difícil de adquirir por la abundancia de mediocridad. Esta escasez se traslada a los formatos de anillos que tenemos. Actualmente (19/02/2026) sólo el formato CODA permite leer y analizar el proceso de cálculo de las simulaciones.
- SETS: Venga, que llegamos a la parte heróica. Llega un momento que, por muy jugoso que sea el anillo (los datos), hay que soltarlo. Esa fuerza interior reside en esta subclase. En ella se nos permite arrojar nuestro granito de arena añadiendo cálculos indirectos al dataset y exportarlos a los formatos disponibles. De nuevo, también está la opción de tirar a Smeagol con el anillo encima (exportar a formato pyLOM), y como en la obra original, esto también se ha dominado para que salga bien.

*ANATOMÍA DE SAM*

Como ya hemos dicho, SAM es ese compañero indispensable de viaje. El mítico compañero de trabajo que habla poco, pero te presenta una solución al apartado que no sabíais hacer ninguno. El callado de clase que misteriosamente saca un 5 en el examen pero que os ha explicado lo más importante de la asignatura para que aprovéis. El héroe sin capa, tan capaz de sostener las emociones del grupo como de enfrentarse a lo orcos más feos. Sus capacidades (clases) se dividen en:
- Gardener: SAM es capaz de organizar todos los datos de FRODO para crear diccionarios de tensores que transformaremos en Datasets. Cocina los datos por nosotros para ponerlos, vengan de donde vengan. También puede aligerarlos (reducir por frecuencia).

- HDF5reader: Un lector básico de archivos .h5 que facilita su estudio.

- Backpack: La mochila mítica del personaje donde encuentras cosas que es mejor llevar y no necesitar, que necesitar y no llevar (como los condones). Aquí encontraremos métodos estáticos que FRODO usará de vez en cuando sin pisparse. Algunos son muy generales y otros específicos, pero pocas veces los tendremos que usar nosotros como usuarios.

- Weapons: La parte más matemática (guerrera) del personaje. Métodos para ordenar geometrías 2D y 3D, métodos de derivación numérica y gradientes y un algoritmo para estudios con GMM. Lo que te daba pereza estudiar, pero que lo necesitas para todo.

- DictVisualizer: La parte más ilustrativa (emocional) del personaje. Como FRODO trabaja mucho con diccionarios, aquí encontramos tres métodos gráficos para resumir la información de estos y que no nos comamos la cabeza con los dramas de FRODO.

En adelante, y como ahora no dispongo de más tiempo, resumiré la ayuda mínima necesaria en comentarios mientras navegas por este notebook. No dudes en escribirme si tienes dudas sobre la convivencia en esta comunidad (miguel.jaraizga@upm.es)