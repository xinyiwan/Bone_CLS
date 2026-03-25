import streamlit as st
import pandas as pd
import pydicom
import numpy as np
import os
from skimage.transform import resize

def pad_image(img, target_height=256, target_width=256):
    """Ajusta la imagen DICOM al tamaño objetivo manteniendo la relación de aspecto."""
    try:
        height, width = img.shape
        scale_factor = min(target_height / height, target_width / width)
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        img_resized = resize(img, (new_height, new_width), anti_aliasing=True)
        
        pad_h = (target_height - new_height) // 2
        pad_w = (target_width - new_width) // 2
        
        padded_img = np.pad(img_resized, 
                            ((pad_h, target_height - new_height - pad_h),
                             (pad_w, target_width - new_width - pad_w)),
                            mode="constant", constant_values=0)
        return padded_img
    except Exception:
        return np.zeros((target_height, target_width))

def load_dicom_dataframe(excel_path):
    """Carga el DataFrame y asegura que existan las columnas de control."""
    try:
        df = pd.read_excel(excel_path)
        if 'seleccion' not in df.columns:
            df['seleccion'] = ""
        if 'viewed' not in df.columns:
            df['viewed'] = ""
        return df
    except Exception as e:
        st.error(f"Error al cargar el Excel: {e}")
        return None
    
    