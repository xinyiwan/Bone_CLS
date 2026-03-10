import os
import pandas as pd
import pydicom
import numpy as np
from utils.Old_classification import *
import re


def cargar_dicom_tags(ruta_dicom,etiquetas_dicom):

    old_class=get_old_classification(ruta_dicom)
    old_class=check_old_class(old_class)
    
    try:
        dicom_medio_path,num_img,num_volumenes,b_value,num_dim,num_dcm,num_echos,Image_type = seleccionar_dicom_intermedio(ruta_dicom, -1)
        valores_etiquetas={}
        # Extraer etiquetas DICOM y guardarlas en el nuevo Excel
        dicom = pydicom.dcmread(dicom_medio_path)
        for etiqueta in etiquetas_dicom:
            val=getattr(dicom, etiqueta, None)
            if etiqueta=='ImageOrientationPatient':
                valores_etiquetas['Plano'],valores_etiquetas['Orientacion'] = analizar_orientacion(val)
            if isinstance(val,pydicom.multival.MultiValue):
                index=0
                for v in val:
                    valores_etiquetas[etiqueta+'_'+str(index+1)] = v
                    index=index+1
            else:
                valores_etiquetas[etiqueta] =val

        # Agregar información al DataFrame
        if None in b_value :
            num_bval=0
            b_value=0
        else:
            num_bval=len(np.unique(b_value))

        try:

            valores_etiquetas.update({"b_value_max":np.max(b_value),"b_value_min":np.min(b_value),"b_value_num":num_bval,"bvalues":b_value,"Num_dcm": num_dcm,"All_image_type":Image_type,
                                        "Num_echos":num_echos,"Num_img": num_img,"Num_vol": num_volumenes,"Old_Class": old_class,"Num_dim": num_dim})
        except:
            valores_etiquetas.update({"b_value_max":0,"b_value_min":0,"b_value_num":0,"bvalues":[],"Num_dcm": num_dcm,"All_image_type":Image_type,
                                        "Num_echos":num_echos,"Num_img": num_img,"Num_vol": num_volumenes,"Old_Class": old_class,"Num_dim": num_dim})
    except Exception as e:
        dicom_medio_path="Error"
        valores_etiquetas={}
    
    df_resultados= pd.DataFrame([valores_etiquetas])

    return df_resultados,dicom_medio_path

    
def get_bvalue_from_name(ds):
    
    name = ds.get(("0018","0024"))
    #numero = re.search(r'\d+', name.value)
    numero = re.search(r'[bB](\d+)', name.value)

    return int(numero.group()[1:]) if numero else None


def get_bvalue(ds):
    tags_bvalue = [("0019", "100c"), ("0043", "1039"), ("2001", "1003"), ("0018", "9087")]
    b_values=[]

    for tag in tags_bvalue:
        b_value = ds.get(tag,None)
        if b_value == None:
            pass
        elif tag == ('0043', '1039'):
            if type(b_value.value)==bytes:
                cadena = b_value.value.decode('utf-8')
                b_value = int(cadena.split('\\')[0])
                b_value = int(b_value) % 100000
                b_values.append(b_value)

            else:
                b_value = int(b_value.value[0]) % 100000
                b_values.append(b_value)

        else:
            b_value = b_value.value
            b_values.append(b_value)

    if len(b_values)>0:
        if np.unique(np.array(b_values))>1:
            return np.max(np.array(b_values))
        else:
            return b_values[0]
    else:
        try:
            b_value=get_bvalue_from_name(ds)
            return b_value
        except:
            return None
 
def analizar_orientacion(image_orientation):

    if image_orientation==None:
            return None,None
    
    # Determinar el plano de adquisición
    row_vector = np.array(image_orientation[:3])
    col_vector = np.array(image_orientation[3:])
    normal_vector = np.cross(row_vector, col_vector)
    dominant_axis = np.argmax(np.abs(normal_vector))

    if dominant_axis == 2:  # Z (superior-inferior)
        plane = "Axial"
    elif dominant_axis == 1:  # Y (anterior-posterior)
        plane = "Coronal"
    elif dominant_axis == 0:  # X (izquierda-derecha)
        plane = "Sagital"
    else:
        plane = "Desconocido"

    # Convertir a string entendible
    axes = ['R', 'A', 'H']  # Right, Anterior, Head
    inverse_axes = ['L', 'P', 'F']  # Left, Posterior, Foot
    
    orientation_str = []
    for i, value in enumerate(image_orientation[:3]):  # Primer vector (filas)
        if abs(value) > 0.5:
            orientation_str.append(axes[i] if value > 0 else inverse_axes[i])
    for i, value in enumerate(image_orientation[3:]):  # Segundo vector (columnas)
        if abs(value) > 0.5:
            orientation_str.append(axes[i] if value > 0 else inverse_axes[i])

    orientation_string = "".join(orientation_str)

    return plane, orientation_string


def seleccionar_dicom_intermedio(carpeta_dicom, volumen=-1):

    # Obtener todos los archivos DICOM
    dicom_files = [os.path.join(carpeta_dicom, f) for f in os.listdir(carpeta_dicom) if f.endswith('.dcm')]
    num_dcm=len(dicom_files)
    if not dicom_files:
        raise ValueError(f"No se encontraron archivos DICOM en {carpeta_dicom}")


    # Extraer metadatos de cada archivo
    positions = []
    series_positions = []
    acquisition_matrix = []
    num_echos=[]
    for file in dicom_files:
        # Leer información del archivo individualmente
        dicom_data = pydicom.dcmread(file, stop_before_pixels=True)

        position = dicom_data[0x0020, 0x0032].value  # Tag '0020|0032'

        try:
            acquisition_matrix.append(dicom_data[0x0018, 0x1310].value)
        except:
            pass
        
        try:
            num_echos.append(dicom_data['EchoNumbers'].value)
        except:
            pass

        # Guardar posición y ruta
        series_positions.append((position, file))

        if position not in positions:
            positions.append(position)  # Convertir a tupla de coordenadas (x, y, z)

        # Resultados: Diccionario con las rutas intermedias por serie

    num_images=len(positions)
    variations = np.ptp(positions, axis=0)
    axis_order = np.argsort(-variations)
    sorted_positions = sorted(positions, key=lambda x: x[axis_order[0]])
    num_dim=len(np.unique(acquisition_matrix, axis=0))
    num_echos= len(np.unique(num_echos))

    # Calcular la posición intermedia
    middle_index = len(positions) // 2
    middle_position = sorted_positions[middle_index]

    middle_slices=([dicom_path for pos, dicom_path in series_positions if pos == middle_position])

    num_volumes = len(middle_slices)
    # Si es multidinámica, seleccionar el volumen

    b_value=[]
    AcquisitionTime=[]
    ContentTime=[]
    Image_type=[]
    if num_volumes==1:
        dicom = pydicom.dcmread(middle_slices[0])
        try:
            b_value.append(get_bvalue(dicom))
        except:
            pass
        try:
            Image_type.append(dicom["ImageType"].value)
        except:
            pass
        dicom_file_medio=middle_slices[0]

    elif num_volumes > 1:
        for middle_slice in middle_slices:
            dicom = pydicom.dcmread(middle_slice)
            try:
                b_value.append(get_bvalue(dicom))
            except:
                pass
            try:
                Image_type.append(dicom["ImageType"].value)
            except:
                pass
            try:
                AcquisitionTime.append(dicom["AcquisitionTime"].value)
            except:
                pass

            try:
                ContentTime.append(dicom["ContentTime"].value)
            except:
                pass

        # Caso 1: Hay más de un b-value
        if len(np.unique([b for b in b_value if b is not None])) > 1:
            sorted_indices = np.argsort(b_value)
            selected_index = sorted_indices[int(len(sorted_indices) // 2)]

        # Caso 2: Un único b-value o ninguno
        elif len(AcquisitionTime) == len(middle_slices) and None not in AcquisitionTime:
            sorted_indices = np.argsort(AcquisitionTime)
            selected_index = sorted_indices[len(sorted_indices) // 2]

        elif len(ContentTime) == len(middle_slices) and None not in ContentTime:
            sorted_indices = np.argsort(ContentTime)
            selected_index = sorted_indices[len(sorted_indices) // 2]

        else:
            # Prioridad: IP/INPHASE > WATER/W > OUT_PHASE/OP > FAT/F
            priorities = ["IP", "INPHASE", "IN_PHASE", "WATER", "W", "OUT_PHASE", "OUTPHASE","OP","OPP_PHASE", "FAT", "F","M"]
            selected_index = None
            for priority in priorities:
                for i, img_type in enumerate(Image_type):
                    if any(priority in str(item).upper() for item in img_type):
                        selected_index = i
                        break
                if selected_index is not None:
                    break

            # Si no se encuentra ningún ImageType prioritario, seleccionar cualquier middle slice
            if selected_index is None:
                selected_index = len(middle_slices) // 2

        
        dicom_file_medio=middle_slices[selected_index]

    return dicom_file_medio,num_images,num_volumes,b_value,num_dim,num_dcm,num_echos,Image_type



def check_old_class(old_class):

    if check_old_class=="IVIM":
        old_class="DW"
    elif check_old_class=="DWI":
        old_class="DW"

    return old_class


