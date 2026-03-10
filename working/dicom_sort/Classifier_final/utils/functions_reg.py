import os
import numpy as np
import os
import pydicom
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# ============================
# FUNCIONES AUXILIARES
# ========================


def load_dicoms_from_folder(folder_path):
    """Carga los archivos DICOM desde una carpeta."""
    dicoms = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            dicom_data = pydicom.dcmread(file_path)
            dicoms.append(dicom_data)
        except:
            continue
    return dicoms

def correct_orientation(dicom, normal):
    """Corrige la orientación de la imagen según la dirección de adquisición."""
    pixel_array = dicom.pixel_array
    
    # Verificar la orientación con el eje normal
    if normal is not None:
        if normal[2] < 0:  # Si el eje Z es negativo, invertir el orden de los slices
            pixel_array = np.flipud(pixel_array)
        if normal[1] < 0:  # Si el eje Y es negativo, invertir verticalmente
            pixel_array = np.fliplr(pixel_array)
    
    return pixel_array


def get_acquisition_axis(dicoms):
    """Determina el eje de adquisición basado en ImageOrientationPatient."""
    
    flag=True
    cont=0
    while flag:
        dicom=dicoms[cont]
        try:
            iop = dicom.ImageOrientationPatient
            row_cosine = np.array(iop[:3])  # Dirección de las filas
            col_cosine = np.array(iop[3:])  # Dirección de las columnas
            
            normal = np.cross(row_cosine, col_cosine)  # Normal al plano de la imagen
            
            # Determinar qué eje es el más dominante
            axis_labels = ["Sagittal", "Coronal", "Axial"]
            main_axis = np.argmax(np.abs(normal))  # Índice del eje dominante (X, Y o Z)
            flag=False
            return axis_labels[main_axis], normal  # Devuelve el eje dominante y la normal
        
        except:
            cont=cont+1
            if cont==len(dicoms):
                return "Unknown", None
    

def sort_dicoms(dicoms):
    """Ordena los cortes DICOM según su posición en el eje correcto."""
    if not dicoms:
        return []

    acquisition_axis, normal = get_acquisition_axis(dicoms)
    
    # Determinar el índice correcto en ImagePositionPatient según el eje de adquisición
    axis_index = {"Sagittal": 0, "Coronal": 1, "Axial": 2}.get(acquisition_axis, 2)
    
    # Ordenar por la posición en el eje de adquisición
    try:
        dicoms_sorted = sorted(dicoms, key=lambda d: float(d.ImagePositionPatient[axis_index]))
        dicoms_with_position=dicoms
    except:
        dicoms_with_position = [d for d in dicoms if 'ImagePositionPatient' in d]
        dicoms_sorted = sorted(dicoms_with_position, key=lambda d: float(d.ImagePositionPatient[axis_index]))
    dicoms_sorted = [correct_orientation(d, normal) for d in dicoms_sorted]

    return dicoms_sorted, acquisition_axis,dicoms_with_position

def extract_coronal_slice(dicoms):
    """Extrae un corte coronal real desde la pila de imágenes DICOM."""
    dicoms_sorted, acquisition_axis,dicoms_with_position = sort_dicoms(dicoms)

    if not dicoms_sorted:
        print("No se encontraron imágenes DICOM válidas.")
        return None

    # Convertir a volumen 3D
    volume = np.stack([d for d in dicoms_sorted], axis=0)  # (Slices, Height, Width)

    # Extraer el corte coronal correcto
    if acquisition_axis == "Axial":
        # Si las imágenes son axiales, el eje Y (altura) está en la segunda dimensión
        slice_idx = volume.shape[1] // 2  # Tomar la mitad del eje Y (altura)
        coronal_slice = volume[:, slice_idx, :]  # Extraer corte coronal
    elif acquisition_axis == "Sagittal":
        # Si las imágenes son sagitales, hay que tomar un corte en el eje X (ancho)
        slice_idx = volume.shape[2] // 2  # Tomar la mitad del eje Z (profundidad)
        coronal_slice = volume[:, :, slice_idx].T  # Transponer para alineación correcta
    elif acquisition_axis == "Coronal":
        # Si ya son coronal, simplemente tomar un slice en el centro
        slice_idx = volume.shape[0] // 2  # Tomar la mitad de los cortes
        coronal_slice = volume[slice_idx, :, :]  # Extraer el corte

    return coronal_slice,acquisition_axis,dicoms_with_position

def get_voxel_size(dicom, acquisition_axis):
    """Obtiene el tamaño del vóxel según el eje de adquisición."""
    try:
        pixel_spacing = dicom.PixelSpacing  # (espaciado en X, espaciado en Y)
        slice_thickness = dicom.SliceThickness  # Espesor de corte (en Z)

        if acquisition_axis == "Axial":
            voxel_size =  (pixel_spacing[0], slice_thickness)  # (X, Y)
        elif acquisition_axis == "Coronal":
            voxel_size = (pixel_spacing[0], pixel_spacing[1]) # (X, Z)
        elif acquisition_axis == "Sagittal":
            voxel_size = (slice_thickness, pixel_spacing[1])  # (Z, Y)
        else:
            voxel_size = (1, 1)  # Valor por defecto si no se encuentra info

        return voxel_size
    except AttributeError:
        print("No se pudo obtener el tamaño del vóxel, usando valores por defecto.")
        return (1, 1)  # Valor por defecto en caso de error


def save_coronal_slice(image, voxel_size,size):
    """Guarda la imagen en formato JPG, escalando según el tamaño del vóxel 
       para que 300 píxeles correspondan a 150 cm (2 píxeles por cm)."""


    # Normalizar la imagen
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    img = Image.fromarray(image)

    # Dimensiones originales en píxeles
    width_px, height_px = img.size

    # Factores de escala (2 píxeles por cm → 300 píxeles = 150 cm)
    target_cm = size  # Queremos que la imagen final represente 150 cm en ambas dimensiones
    pixels_per_cm = 300 / target_cm  # 2 píxeles por cm

    # Escalamos la imagen para que se ajuste a 150 cm en el lado más largo
    scale_x = pixels_per_cm * voxel_size[0]  # Factor de escala en X
    scale_y = pixels_per_cm * voxel_size[1]  # Factor de escala en Y

    # Nuevas dimensiones de la imagen en píxeles
    new_width = int(width_px * scale_x)
    new_height = int(height_px * scale_y)

    # Redimensionar la imagen con interpolación
    img_rescaled = img.resize((new_width, new_height), Image.LANCZOS)

    # Crear una imagen de fondo negro de 300x300
    background = Image.new("L", (300, 300), 0)  # "L" = escala de grises, 0 = negro

    # Si la imagen reescalada es más grande que 300x300, la reducimos para que entre
    if new_width > 300 or new_height > 300:
        scale_factor = min(300 / new_width, 300 / new_height)  # Escalar manteniendo relación
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)
        img_rescaled = img_rescaled.resize((new_width, new_height), Image.LANCZOS)

    # Calcular posición para centrar la imagen en el fondo negro
    x_offset = (300 - new_width) // 2
    y_offset = (300 - new_height) // 2

    # Pegar la imagen escalada sobre el fondo negro
    background.paste(img_rescaled, (x_offset, y_offset))

    return background

def load_img_C( ruta_dicom,size=300):

    try:       
        dicoms = load_dicoms_from_folder(ruta_dicom)
        coronal_slice, acquisition_axis,dicoms_sort= extract_coronal_slice(dicoms)           
        voxel_size = get_voxel_size(dicoms_sort[0], acquisition_axis)  
        img=save_coronal_slice(coronal_slice, voxel_size,size)
        
    except Exception as e:
        img="Error"

    img = img_to_array(img)        # Convertimos a array
    img = np.expand_dims(img, axis=0) 

    return img


def check_dicom_tags(dicom_tags_p, Error):
    for tag in dicom_tags_p.columns:
        # Comprobar si la columna existe y contiene algún valor nulo o vacío
        if dicom_tags_p[tag].isnull().any() or (dicom_tags_p[tag] == '').any():
            dicom_tags_p[tag] = dicom_tags_p[tag].fillna(0).replace('', 0)
            Error.append(f"Dicom Tag {tag} not found or null")
        
    if Error==[]:
        Error=""
    
    return dicom_tags_p, Error