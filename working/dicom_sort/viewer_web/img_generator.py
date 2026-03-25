import os
import json
import pandas as pd
import pickle
import logging
import traceback 
from tqdm import tqdm
import numpy as np
import argparse
import concurrent.futures
from functools import partial
import pydicom
from PIL import Image

def save_img(imgs_folder, path_serie):
    try:
        # Extraer Paciente, Estudio, Serie de la ruta (asumiendo que las carpetas están al final)
        # Nota: He ajustado esto asumiendo que path_serie es la ruta a la carpeta de la serie
        patient, study, serie = path_serie.split(os.sep)[-4:-1]
        img_path = os.path.join(imgs_folder, patient, study, serie)
        
        ds = pydicom.dcmread(path_serie)
        data = ds.pixel_array
        
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        data = data.astype(np.float32) * slope + intercept

        val_low = np.percentile(data, 1)
        val_high = np.percentile(data, 99)
        data_clipped = np.clip(data, val_low, val_high)

        if val_high != val_low:
            data_norm = (data_clipped - val_low) / (val_high - val_low)
            img_uint8 = (data_norm * 255).astype(np.uint8)
        else:
            img_uint8 = np.zeros(data.shape, dtype=np.uint8)
        
        os.makedirs(img_path, exist_ok=True)
        img_file = os.path.join(img_path, "Img.png")
        Image.fromarray(img_uint8).save(img_file)
        
        return {"status": "success", "path": img_file}
        
    except Exception as e:
        logging.error(f"Error procesando {path_serie}: {e}")
        return None


def main():
    # --- 1. CAPTURAR ARGUMENTOS DESDE STREAMLIT ---
    parser = argparse.ArgumentParser(description="Generate PNG previews from DICOM files")
    parser.add_argument("--excel",   required=True, help="Path to classifier CSV or Excel file")
    parser.add_argument("--out_dir", required=True, help="Base directory for saving generated images")
    args = parser.parse_args()

    # Asignamos las variables basándonos en los argumentos del parser
    path_excel = args.excel
    path_jpg = args.out_dir

    # --- 2. LEER CONFIGURACIÓN (Solo para los nombres de las columnas) ---
    path_project = os.environ.get('INPUT_PATH', "/Proyecto")
    config = os.environ.get('CONFIG_PATH', "/Parameters_config")
    config_filename = os.environ.get('CONFIG_FILE', 'parameter_configuration.json')
    config_file = os.path.join(config, config_filename)

    try:
        with open(config_file, 'r') as f:
            params_config = json.load(f)
        data = params_config.get('VIEWER', {})
    except Exception as e:
        print(f"Aviso: No se pudo cargar el JSON de configuración ({e}). Se usarán valores por defecto.")
        data = {}

    # Si tu Excel desde Streamlit tiene columnas fijas, ponlas aquí por defecto
    patient_column = data.get('patient_column', 'Paciente')
    study_column = data.get('study_column', 'Estudio')
    serie_column = data.get('serie_column', 'Serie')
    dicom_column = data.get('dicom_column', 'DICOM') # o 'DICOM'

    print(f"Loading file: {path_excel}")
    if path_excel.lower().endswith(".csv"):
        df_datos = pd.read_csv(path_excel, dtype=str).fillna("")
    else:
        df_datos = pd.read_excel(path_excel)

    for col in [patient_column, study_column, serie_column]:
        if col not in df_datos.columns:
            raise ValueError(f"Column '{col}' not found. Available columns: {df_datos.columns.tolist()}")

    df_datos['ruta_completa'] = df_datos.apply(
        lambda row: os.path.join(
            path_project, 
            str(row[patient_column]), 
            str(row[study_column]), 
            str(row[serie_column]),
            str(row[dicom_column])
        ), 
        axis=1
    )

    rutas_a_procesar = df_datos['ruta_completa'].unique().tolist()
    print(f"Unique DICOM paths to process: {len(rutas_a_procesar)}")

    worker_func = partial(save_img, path_jpg)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker_func, path): path for path in rutas_a_procesar}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(rutas_a_procesar), desc="Processing DICOM"):
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception as exc:
                print(f"Worker failed: {exc}")

    print("Image generation complete.")

if __name__ == "__main__":
    main()