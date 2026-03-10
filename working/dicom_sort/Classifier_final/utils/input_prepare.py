import numpy as np
import tensorflow as tf
import pydicom
import cv2


def cargar_entrada(dicom_img,dicom_tags_p,Variables_dicom,img_size,net,scaler):


    imagenes = read_img(dicom_img,img_size,net)  
    imagenes = np.expand_dims(imagenes, axis=0)  # Si falta una dimensión de batch

    datos_numericos = dicom_tags_p[Variables_dicom].copy()
    mascara_datos_numericos = datos_numericos.notna().astype(int)
    datos_numericos = datos_numericos.astype(float).values
    mascara_datos_numericos = mascara_datos_numericos.values  # Asegura que sea un array numpy
    datos_numericos = np.nan_to_num(datos_numericos, nan=0.0)

    datos_numericos = scaler.fit_transform(np.hstack([datos_numericos]))

    
    return imagenes,datos_numericos,mascara_datos_numericos

def cargar_entrada_img(ruta_datos,img_size,net):
    imagenes = read_img(ruta_datos,img_size,net)  
    imagenes = np.expand_dims(imagenes, axis=0)  # Si falta una dimensión de batch

    return imagenes


def read_img(ruta_dicom,img_size,net):

    dicom = pydicom.dcmread(ruta_dicom)

    if dicom.file_meta.TransferSyntaxUID.is_compressed:
        dicom.decompress()

    img = dicom.pixel_array
    img = cv2.resize(img, img_size)

    if net.startswith('EfficientNet'):
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = img.astype(np.float32)
    elif net.startswith('ConvNeX'):
        img = tf.keras.applications.convnext.preprocess_input(img)
        img = img.astype(np.float32)
    elif net.startswith('ResNet'):
        img = tf.keras.applications.resnet50.preprocess_input(img)
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32) / np.max(img)

    return img