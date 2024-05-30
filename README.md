# Descripción
Este programa permite detectar el polvo en una imágen. 
Las imágenes son redimensionadas, tratadas por el modelo para conseguir polvo y añadidas a una cuadrícula de análisis de densidad de polvo.

El programa saca tres resultados:
*   Las imágenes originales redimensionadas
*   Las imágenes con polvo detectado
*   La cuadrícula (cuadro de ajedrez) con el análisis de densidad de polvo 

# Proceso

* Clonar el repositorio.
* Crear un ambiente en conda con el python 3.10.14
* Descargar los archivos de requirements.txt
* Correr el comando: python src/model.py

# Resultados
<p>
<h2>Original
<br>
<img src="output/original/1_original.png" alt="drawing" width="350" height="350"/>
<br>
Polvo
<br>
<img src="output/polvo/1_polvo.png" alt="drawing" width="350" height="350"/>
<br>
Ajedrez</h2>
<br>
<img src="output/ajedrez/1_ajedrez.png" alt="drawing" width="350" height="350"/>
</p>