# MAPAS ADC
@Author: __Matias fernandez Paton__

## Docker MAPAS ADC

Código para clasificar las secuencias segun su ponderación, supresion grasa, contraste y familia; separando los casos que no son usados para analisis (Mapas, localizadores...) a partir de un excel de entrada con las rutas a las series que se quieren clasificar o una carpeta con los pacientes que se quieren clasificar. El resultado es un excel con las rutas y algunas de las caracteristicas de cada imagen.

### Requerimientos previos

La estructura de carpetas tiene que ser la siguiente:

- PROYECTO
	- ADQUISICIONES
		- PACIENTE
			- ESTUDIO
					- SERIE
						- File.dcm
						- File1.dcm
						- ...
						- Filen.dcm
		
				- SEGMENTATION 
					- SERIE
						- Mask.nii
					

## Code execution

1. Abrir Docker Desktop
2. Abrir una nueva consola

### Primeros pasos: construcción de la imagen docker

Esta sección solo se tiene que ejecutar una vez.
Para comprobar si ya existe la imagen docker ejecutar la siguiente línea:

~~~~
  docker images
~~~~

Si la imagen ya existe, saltar pasos 3 y 4. 

3. Situarse en la carpeta donde se encuentra la carpeta Clasificador_seq usando el siguiente comando:

~~~~
  cd "C:\path\of\Clasificador_seq"
~~~~

donde C:\path\of\Clasificador_seq  es la ruta a la carpeta Clasificador_seq 

4. Construcción del docker con el siguiente comando:

~~~~
  docker build -f .\docker\Dockerfile -t "dockername":"tag" .
~~~~

donde:

* **dockername** nombre que se le asigna a la imagen docker (en minúsculas)
* **tag** versión de la imagen docker (en minúsculas o números)


### Ejecución imagen docker

~~~~
  docker run --rm --env-file=.env -v C:\path\of\Project:/Proyecto  -v C:\path\of\Parameterfolder:/Parameters_config  -v C:\path\of\Path_excel:/Path_excel -t dockername:tag 
~~~~

donde : 

* **C:\path\of\Project** ruta del proyecto
* **C:\path\of\Parameterfolder** ruta de la carpeta donde se encuentra el fichero de configuración de parámetros
* **C:\Path_excel** (opcional) ruta a la carpeta donde se encuentra el excel con lkas rutas de las series a clasificar
* **dockername** es el nombre de la imagen docker 
* **tag** versión de la imagen docker

#### Observaciones
Se debe consultar el contenido de **.env** para saber los directorios internos que se deben especificar al crear los volúmenes en el comando *docker run*. Además, el usuario podrá modificar estas rutas según sus preferencias, así como el nombre de los archivos de configuración. Por defecto, se leerá *paramter_configuration.json*.


### Fichero de configuración de parámetros .json

El fichero puede llamarse parameter_configuration.json (u otro nombre, siempre que se modifique en el **.env**) y debe encontrase en la carpeta "C:\path\of\Parameterfolder". Debe tener el siguiente contenido:

~~~~
{ 
	"CLASSIFIER": {
		"excel_flag":true,	
		"input_folder": "DCM",
		"excel_path":"/Proyecto/Training_excels/Test_final.xlsx",
		"excel_column":"Path"
		}
}
~~~~

donde : 

* **excel_flag**  indica si la calsificación se va a realizar a traves de un excel (true) o si por el contrario se van a clasificar todos los pacientes de la carpeta "input_folder" (false)
* **input_folder** carpeta con los pacientes a clasificar, excel_flag tiene que ser false
* **excel_path** ruta al excel con las rutas a las series a clasificar , excel _flag tiene que ser true
* **excel_column** nombre de la columna donde se encuentran las rutas.

En la carpeta *Parameters_config*  hay un ejemplo de fichero de configuración de parámetros.


### Estructura guardado

A nivel de /Proyecto se generara una carpeta llamada Resultados y dentro un excel con las rutas y las anteriores con las caracteristicas de cada serie.


Contact: <matias_fernandez@iislafe.es>




