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
    parser = argparse.ArgumentParser(description="Generar imágenes PNG desde DICOM")
    parser.add_argument("--excel", required=True, help="Ruta al archivo Excel temporal")
    parser.add_argument("--out_dir", required=True, help="Ruta base para guardar las imágenes generadas")
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

    print(f"Cargando Excel desde: {path_excel}")
    df_datos = pd.read_excel(path_excel)

    # Validar que existan las columnas en el Excel
    for col in [patient_column, study_column, serie_column]:
        if col not in df_datos.columns:
            raise ValueError(f"La columna '{col}' no existe en el Excel. Columnas encontradas: {df_datos.columns.tolist()}")

    # 3. RECOLECTAR TODAS LAS RUTAS A PROCESAR
    # (Construimos la ruta base de la serie, ajusta si tu dicom_column incluye el nombre del archivo)
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

    # Obtener la lista de rutas únicas para no procesar la misma carpeta dos veces
    rutas_a_procesar = df_datos['ruta_completa'].unique().tolist()
    print(f"Total de carpetas DICOM únicas a procesar: {len(rutas_a_procesar)}")

    # 4. CONFIGURAR LA FUNCIÓN PARCIAL
    # Al poner imgs_folder como primer argumento en save_img, partial lo "congela"
    # y los workers solo le enviarán el 'path' como segundo argumento.
    worker_func = partial(save_img, path_jpg)

    resultados_totales = []

    # 5. LANZAR EL PROCESAMIENTO EN PARALELO
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker_func, path): path for path in rutas_a_procesar}
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(rutas_a_procesar), desc="Procesando MRI"), 1):
            try:
                resultado_dict = future.result()
                if resultado_dict:
                    resultados_totales.append(resultado_dict)
            except Exception as exc:
                print(f"Fallo en hilo: {exc}")

    print("¡Proceso de generación finalizado!")

if __name__ == "__main__":
    main()