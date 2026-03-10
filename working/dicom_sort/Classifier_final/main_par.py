import os
import json
import pandas as pd
from utils.dicom_tag_preprocess import *
from utils.dicom_tag_load import *
from utils.input_prepare import *
from utils.Models import *
from utils.Models_reg import *
from utils.functions_reg import *
import pickle
import logging
import traceback 
from tqdm import tqdm
import numpy as np
import argparse
import concurrent.futures
from functools import partial


def load_existing_results(csv_path_out):
    """Load existing results from CSV file if it exists"""
    if os.path.exists(csv_path_out):
        try:
            df_existing = pd.read_csv(csv_path_out)
            # Create a set of tuples (Paciente, Estudio, Serie) for already processed series
            processed_series = set()
            for _, row in df_existing.iterrows():
                paciente = str(row.get('Paciente', '')) if pd.notna(row.get('Paciente', '')) else ''
                estudio = str(row.get('Estudio', '')) if pd.notna(row.get('Estudio', '')) else ''
                serie = str(row.get('Serie', '')) if pd.notna(row.get('Serie', '')) else ''
                
                if paciente and estudio and serie:
                    processed_series.add((paciente, estudio, serie))
            
            resultados = df_existing.to_dict('records')
            logging.info(f"Loaded {len(processed_series)} existing results from {csv_path_out}")
            return resultados, processed_series
        except Exception as e:
            logging.warning(f"Could not load existing results file: {e}")
            return [], set()
    else:
        logging.info("No existing results file found, starting fresh")
        return [], set()

def is_series_processed(patient, study, serie, processed_series):
    """Check if a series has already been processed"""
    return (str(serie), str(study), str(patient)) in processed_series

def save_results_incrementally(resultados, csv_path_out, is_final=False):
    """Save results to CSV file"""
    try:
        df = pd.DataFrame(resultados)
        df.to_csv(csv_path_out, index=False)
        save_type = "Final" if is_final else "Checkpoint"
        logging.info(f"{save_type} results saved to {csv_path_out} ({len(resultados)} total records)")
        return True
    except Exception as e:
        logging.error(f"Error saving results to CSV: {str(e)}")
        # Emergency backup with timestamp
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = csv_path_out.replace('.csv', f'_backup_{timestamp}.csv')
            pd.DataFrame(resultados).to_csv(backup_path, index=False)
            logging.info(f"Emergency backup saved to {backup_path}")
            return True
        except:
            return False

def extract_path_components(path_serie):
    """Extract patient, study, serie from path"""
    try:
        path_aux, serie = os.path.split(path_serie)
        path_aux, study = os.path.split(path_aux)
        _, patient = os.path.split(path_aux)
        return patient, study, serie
    except:
        return "", "", ""

def classifier(model_other, dicom_tags_others, model_weighting, dicom_tags_weigthing, model_fs, dicom_tags_fs, 
               model_family, dicom_tags_family, model_c, model, Etiquetas_dicom, Img_size, Model_name, path_serie):
    """
    Recibe la ruta y los modelos, devuelve un diccionario con el resultado.
    """
    logging.info(f"Procesando: {path_serie}")
    label_A=['2D','3D']
    label_family= ["EP","GR","IR","SE"]

    path_aux, patient = os.path.split(path_serie)
    path_aux, study = os.path.split(path_aux)
    _, serie = os.path.split(path_aux)
    
    Error = []
    dicom_img = "Unknown"

    try:
        logging.info('Starting Loading Dicom Tags')
        dicom_tags, dicom_img = cargar_dicom_tags(path_serie, Etiquetas_dicom)

        logging.info('Dicom: %s', dicom_img)

        logging.info('Starting Preprocessing Dicom Tags')
        dicom_tags_p = process_dicom_tags(dicom_tags)

        dicom_tags_p, err_check = check_dicom_tags(dicom_tags_p, [])
        if err_check:
            Error.append(err_check[0])

        logging.info('Starting Prediction')

        Num_dim = obtener_valor(dicom_tags_p, 'Num_dim', 0)
        Num_img = obtener_valor(dicom_tags_p, 'Num_img', 0)
        adquisition_dimension = obtener_valor(dicom_tags_p, 'MRAcquisitionType', '-')
        
        # Convertir etiquetas si los valores son válidos
        try:
            if adquisition_dimension not in ['N/A', '-']:
                try:
                    idx = int(adquisition_dimension)
                    if 0 <= idx < len(label_A):
                        adquisition_dimension = label_A[idx]
                    else:
                        adquisition_dimension = 'Unknown'
                except (ValueError, TypeError):
                    adquisition_dimension = 'Unknown'
            else:
                adquisition_dimension = 'Unknown'
        except Exception as e:
            logging.error(f"Error en Adquisition Dimension: {e}")
            adquisition_dimension = 'Unknown'

        if Num_dim > 1 or Num_img < 10:
            pred_others = False
            pred_weigthing_p = 1
        else:
            try:    
                pred_others = model_other.predict(dicom_tags_p[dicom_tags_others].to_numpy()[0].reshape(1, -1))
                pred_weigthing_p = np.max(model_other.predict_proba(dicom_tags_p[dicom_tags_others].to_numpy()[0].reshape(1, -1)))
            except Exception as e:
                logging.error(f"Error en Scanning Sequence Family: {e}")
                pred_others = None
                pred_weigthing_p = 1

        # Initialize variables with defaults
        predicciones_clases_h = predicciones_clases_n = predicciones_clases_t = predicciones_clases_a = [["-"]]
        predicciones_clases_p = predicciones_clases_ll = predicciones_clases_s = predicciones_clases_ul = [["-"]]
        predicciones_clases_w = ["Unknown"]
        predicciones_clases_c = "-"
        predicciones_clases_fs = "-"
        predicciones_clases_f = "Unknown"
        pred_c_p = [1]
        pred_fs_p = 1
        predicciones_clases_f_p = 1

        if pred_others == True:
            #### REGION CLASSIFIER #######
            try:
                logging.info('Starting Loading Dicom Tags')
                img = load_img_C(path_serie, 300)

                predicciones = model.predict(img)
                predicciones_clases_h = np.round(predicciones[0])
                predicciones_clases_n = np.round(predicciones[1])
                predicciones_clases_t = np.round(predicciones[2])
                predicciones_clases_a = np.round(predicciones[3])
                predicciones_clases_p = np.round(predicciones[4])
                predicciones_clases_ll = np.round(predicciones[5])
                predicciones_clases_s = np.round(predicciones[6])
                predicciones_clases_ul = np.round(predicciones[7])

            except Exception as e:
                if Num_dim > 1: 
                    logging.info("Survey or Localizer")
                    predicciones_clases_h = predicciones_clases_n = predicciones_clases_t = predicciones_clases_a = [["-"]]
                    predicciones_clases_p = predicciones_clases_ll = predicciones_clases_s = predicciones_clases_ul = [["-"]]
                    Error.append("Survey or Localizer")
                else:
                    predicciones_clases_h = predicciones_clases_n = predicciones_clases_t = predicciones_clases_a = [[None]]
                    predicciones_clases_p = predicciones_clases_ll = predicciones_clases_s = predicciones_clases_ul = [[None]]
                    Error.append(str(e))

            try:
                pred_weigthing = model_weighting.predict(dicom_tags_p[dicom_tags_weigthing].to_numpy()[0].reshape(1, -1))
                pred_weigthing_p = np.max(model_weighting.predict_proba(dicom_tags_p[dicom_tags_weigthing].to_numpy()[0].reshape(1, -1)))
                predicciones_clases_w = pred_weigthing[0]
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                predicciones_clases_w = ["Unknown"]
                pred_weigthing_p = 1

            if predicciones_clases_w in ['T2W', 'T1W']:
                try:
                    pred_fs = model_fs.predict(dicom_tags_p[dicom_tags_fs].to_numpy()[0].reshape(1, -1))
                    pred_fs_p = np.max(model_fs.predict_proba(dicom_tags_p[dicom_tags_fs].to_numpy()[0].reshape(1, -1)))
                    predicciones_clases_fs = 'Y' if pred_fs[0] == True else 'N'    
                except Exception as e:
                    logging.error(f"Error in prediction: {e}")
                    predicciones_clases_fs = None  

                logging.info('Starting Loading input')

                if predicciones_clases_w == 'T1W':
                    try:
                        img = cargar_entrada_img(dicom_img, Img_size, Model_name)
                        pred_c = model_c.predict([img])
                        pred_c_p = pred_c[0] if np.round(pred_c[0]) == 1 else 1 - pred_c[0]   
                        predicciones_clases_c = 'Y' if np.round(pred_c[0]) == 1 else 'N'    
                    except Exception as e:
                        logging.error(f"Error in prediction: {e}")
                        predicciones_clases_c = None   
                
                logging.info('Prediction Done')

            try:
                predicciones_clases_f = scanning_sequence(dicom_tags)
            except Exception as e:
                logging.error(f"Error en Scanning Sequence: {e}")
                predicciones_clases_f = 'Unknown'
                predicciones_clases_f_p = 1

            if predicciones_clases_f in ["Unknown", "RM"]:
                try:
                    pred_family = model_family.predict(dicom_tags_p[dicom_tags_family].to_numpy()[0].reshape(1, -1))
                    predicciones_clases_f_p = np.max(model_family.predict_proba(dicom_tags_p[dicom_tags_family].to_numpy()[0].reshape(1, -1)))
                    predicciones_clases_f = label_family[np.squeeze(pred_family)]
                except Exception as e:
                    logging.error(f"Error en Scanning Sequence Family: {e}")
                    predicciones_clases_f = 'Unknown'       

        elif pred_others == False or pred_others is None:
            if Num_dim > 1 or Num_img < 10: 
                logging.info("Survey or Localizer")
                Error.append("Survey or Localizer")

            if pred_others == False or dicom_img == "Error":
                predicciones_clases_w = ['Other']
                predicciones_clases_fs = '-'    
                predicciones_clases_c = '-'
                predicciones_clases_f = '-'
            else:
                predicciones_clases_w = ["Error"]
                predicciones_clases_fs = None   
                predicciones_clases_c = None
                predicciones_clases_f = None
                pred_weigthing_p = 1

        logging.info('Prediction Completed')

    except Exception as e:
        error_msg = f"Error en Paciente: {patient}, Estudio: {study}, Serie: {serie}, DICOM: {dicom_img}\n{traceback.format_exc()}"
        logging.error(error_msg)
        pred_weigthing_p = 1
        pred_c_p = [1]
        pred_fs_p = 1
        predicciones_clases_f_p = 1
        predicciones_clases_h = predicciones_clases_n = predicciones_clases_t = predicciones_clases_a = [[None]]
        predicciones_clases_p = predicciones_clases_ll = predicciones_clases_s = predicciones_clases_ul = [[None]]
        Error.append(str(e))
        
        if dicom_img == "Error":
            predicciones_clases_w = ['Other']
            predicciones_clases_fs = '-'
            predicciones_clases_c = '-'
            predicciones_clases_f = '-'
        else:
            predicciones_clases_w = ["Error"]
            predicciones_clases_fs = 'Error'
            predicciones_clases_c = 'Error'
            predicciones_clases_f = 'Error'

    # Return dictionary
    return {
        "Paciente": patient,
        "Estudio": study,
        "Serie": serie,
        "Nombre DICOM": dicom_img,  
        "Predicción Clases W": predicciones_clases_w[0] if isinstance(predicciones_clases_w, (list, np.ndarray)) else predicciones_clases_w,
        "Predicción Clases W P": pred_weigthing_p,
        "Predicción Clases FS": predicciones_clases_fs,  
        "Predicción Clases FS P": pred_fs_p,  
        "Predicción Clases C": predicciones_clases_c,  
        "Predicción Clases C P": pred_c_p[0] if isinstance(pred_c_p, (list, np.ndarray)) else pred_c_p,  
        "Scanning Sequence": predicciones_clases_f,
        "Scanning Sequence P": predicciones_clases_f_p,
        "Num Bval": obtener_valor(dicom_tags_p, 'b_value_num', '-'),
        "B-values": obtener_valor(dicom_tags_p, 'bvalues', '-'),
        "Num dim": Num_dim,        
        "Num Vol": obtener_valor(dicom_tags_p, 'Num_vol', '-'),
        "Num Cortes": Num_img ,
        "Num_dcm": obtener_valor(dicom_tags_p, 'Num_dcm', '-'),
        "Num_echos": obtener_valor(dicom_tags_p, 'EchoNumbers', 0),
        "Inversion_time": obtener_valor(dicom_tags_p, 'InversionTime', '-'),
        "Adquisition Dimension": adquisition_dimension,
        "Plano": obtener_valor(dicom_tags_p, 'Plano', '-'),
        "Orientación": obtener_valor(dicom_tags_p, 'Orientacion', '-'),
        "Manufacturer": obtener_valor(dicom_tags_p, 'Manufacturer', 'Unknown'),
        "ManufacturerModelName": obtener_valor(dicom_tags_p, 'ManufacturerModelName', 'Unknown'),
        "MagneticFieldStrength": obtener_valor(dicom_tags_p, 'MagneticFieldStrength', 'Unknown'),
        "Fat Suppresion": obtener_valor(dicom_tags_p, 'F', '-'),
        "Water Suppresion": obtener_valor(dicom_tags_p, 'W', '-'),
        "In Phase": obtener_valor(dicom_tags_p, 'IP', '-'),
        "Out Phase": obtener_valor(dicom_tags_p, 'OP', '-'),
        "Real": obtener_valor(dicom_tags_p, 'ImageType_R', '-'),
        "Imaginary": obtener_valor(dicom_tags_p, 'ImageType_I', '-'),
        "Magnitude": obtener_valor(dicom_tags_p, 'ImageType_M', '-'),
        "Phase": obtener_valor(dicom_tags_p, 'ImageType_P', '-'),
        "All_image_type": obtener_valor(dicom_tags, 'All_image_type', 'Unknown') if 'dicom_tags' in locals() else 'Unknown',
        "Num ImageType": len(obtener_valor(dicom_tags, 'All_image_type', 'Unknown')) if 'dicom_tags' in locals() else 0,
        "H": predicciones_clases_h[0][0] if predicciones_clases_h and predicciones_clases_h[0] is not None else None,
        "N": predicciones_clases_n[0][0] if predicciones_clases_n and predicciones_clases_n[0] is not None else None,
        "T": predicciones_clases_t[0][0] if predicciones_clases_t and predicciones_clases_t[0] is not None else None,
        "A": predicciones_clases_a[0][0] if predicciones_clases_a and predicciones_clases_a[0] is not None else None,
        "P": predicciones_clases_p[0][0] if predicciones_clases_p and predicciones_clases_p[0] is not None else None,
        "LL": predicciones_clases_ll[0][0] if predicciones_clases_ll and predicciones_clases_ll[0] is not None else None,
        "S": predicciones_clases_s[0][0] if predicciones_clases_s and predicciones_clases_s[0] is not None else None,
        "UL": predicciones_clases_ul[0][0] if predicciones_clases_ul and predicciones_clases_ul[0] is not None else None,
        "Error": ", ".join(Error) if Error else ""
    }


def main():
    parser = argparse.ArgumentParser(description="Clasificador de secuencias")
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all series')
    parser.add_argument('--workers', type=int, default=4, help='Número de workers')
    parser.add_argument('--output', type=str, default='csv', choices=['csv', 'excel'], 
                       help='Formato de salida (csv recomendado)')
    args = parser.parse_args()

    Etiquetas_dicom = ['EchoTime', 'InversionTime',  'PixelSpacing', 'RepetitionTime','SliceThickness','MRAcquisitionType','FlipAngle',
                       'PixelBandwidth','ImageType','SequenceVariant','ScanningSequence','ImageOrientationPatient','EchoTrainLength',
                       'ScanOptions','SpectrallySelectedSuppression','Manufacturer','ManufacturerModelName','MagneticFieldStrength']

    path_project = os.environ.get('INPUT_PATH',"/Project")
    output_folder = os.environ.get('OUTPUT_PATH',"/Output")
    config = os.environ.get('CONFIG_PATH',"/Parameters_config")
    config_filename = os.environ.get('CONFIG_FILE', 'parameter_configuration.json')
    config_file = os.path.join(config, config_filename)

    os.makedirs(os.path.join(output_folder,'Results'), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'Logs'), exist_ok=True)

    # Use CSV by default (more robust)
    if args.output == 'csv':
        output_path = os.path.join(output_folder, 'Results', 'Sequence_Classifier.csv')
    else:
        output_path = os.path.join(output_folder, 'Results', 'Sequence_Classifier.xlsx')
    
    log_path = os.path.join(output_folder,'Logs','Sequence_Classifier.log')

    logging.basicConfig(
        filename=log_path, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info('Starting Sequence Classifier')
    logging.info(f'Force mode: {args.force}')
    logging.info(f'Workers: {args.workers}')
    logging.info(f'Output format: {args.output}')

    # Load existing results
    resultados_totales, processed_series = load_existing_results(output_path)

    logging.info('Loading Parameters file')

    with open(config_file, 'r') as f:
        params_config = json.load(f)

    data = params_config.get('CLASSIFIER', {})
    path_model_folder_config = "/module"

    logging.info('Loading Models Configuration...')
    
    # Load models
    path_model_other = os.path.join(path_model_folder_config,'Models/Model_others.pkl')
    with open(path_model_other, "rb") as archivo: 
        model_other = pickle.load(archivo)
    with open(os.path.join(path_model_folder_config,"Models/Model_others.json"), 'r') as f: 
        dicom_tags_others = json.load(f)['ETIQUETAS_DICOM']

    path_model_weigthing = os.path.join(path_model_folder_config,'Models/Model_weigthing.pkl')
    with open(path_model_weigthing, "rb") as archivo: 
        model_weighting = pickle.load(archivo)
    with open(os.path.join(path_model_folder_config,"Models/Model_weigthing.json"), 'r') as f: 
        dicom_tags_weigthing = json.load(f)['ETIQUETAS_DICOM']

    path_model_family = os.path.join(path_model_folder_config,'Models/Model_family.pkl')
    with open(path_model_family, "rb") as archivo: 
        model_family = pickle.load(archivo)
    with open(os.path.join(path_model_folder_config,"Models/Model_family.json"), 'r') as f: 
        dicom_tags_family = json.load(f)['ETIQUETAS_DICOM']

    path_model_fs = os.path.join(path_model_folder_config,'Models/Model_fs.pkl')
    with open(path_model_fs, "rb") as archivo: 
        model_fs = pickle.load(archivo)
    with open(os.path.join(path_model_folder_config,"Models/Model_fs.json"), 'r') as f: 
        dicom_tags_fs = json.load(f)['ETIQUETAS_DICOM']

    with open(os.path.join(path_model_folder_config,"Models/Model_c.json"), 'r') as f: 
        data_model_c = json.load(f)
    Model_type = data_model_c['MODEL']
    Training_mode = data_model_c['TRAINING_MODE']
    learning_rate = data_model_c['LEARNING_RATE']
    Model_name = data_model_c['MODEL_NAME']
    Img_size = data_model_c['IMG_SIZE']
    model_c = crear_modelo_img(Model_type, Training_mode, ["1"], learning_rate, 'c', Img_size, 'sigmoid', 'binary_crossentropy', Model_name)   
    model_c.load_weights(os.path.join(path_model_folder_config,'Models/Model_c.h5'))

    with open(os.path.join(path_model_folder_config,"Models/Model_region.json"), 'r') as f: 
        data_model_reg = json.load(f)
    Model_type_reg = data_model_reg['MODEL']
    Labels = data_model_reg['LABELS']
    Model_name_reg = data_model_reg['MODEL_NAME']
    Img_size_reg = data_model_reg['IMG_SIZE']
    model = crear_modelo_reg(Model_type_reg, Labels, Img_size_reg, Model_name_reg)   
    model.load_weights(os.path.join(path_model_folder_config,'Models/Model_region.h5'))

    logging.info('Models Loaded. Starting Prediction')

    # Collect paths to process
    rutas_a_procesar = []
    skipped_count = 0
    
    excel_flag = data.get('excel_flag', False)
    if excel_flag:
        excel_path = data.get('excel_path', None)
        excel_column = data.get('excel_column', None)
        if excel_path is None or excel_column is None:
            raise ValueError("Faltan 'excel_path' o 'excel_column' en la configuración.")
        df_datos = pd.read_excel(excel_path)
        all_rutas = df_datos[excel_column].tolist()
        
        # Filter out already processed
        if not args.force:
            for path_serie in all_rutas:
                patient, study, serie = extract_path_components(path_serie)
                if is_series_processed(patient, study, serie, processed_series):
                    skipped_count += 1
                else:
                    rutas_a_procesar.append(path_serie)
        else:
            rutas_a_procesar = all_rutas
            
    else: 
        path_adquisition = os.path.join(path_project, data.get('input_folder', 'ADQUISICIONES'))
        patients = os.listdir(path_adquisition)
        for patient in patients:
            path_patient = os.path.join(path_adquisition, patient)
            if os.path.isdir(path_patient):
                try:
                    studies = os.listdir(path_patient)
                except:
                    continue
                    
                for study in studies:
                    if 'MR' in study:
                        path_study = os.path.join(path_patient, study)
                        try:
                            series = os.listdir(path_study)
                        except:
                            continue
                            
                        for serie in series:
                            path_serie = os.path.join(path_adquisition, patient, study, serie)
                            
                            if not args.force and is_series_processed(patient, study, serie, processed_series):
                                skipped_count += 1
                            else:
                                rutas_a_procesar.append(path_serie)

    print(f"\nTotal series encontradas: {len(rutas_a_procesar) + skipped_count}")
    print(f"Series nuevas a procesar: {len(rutas_a_procesar)}")
    print(f"Series ya procesadas (saltadas): {skipped_count}\n")

    if not rutas_a_procesar:
        print("No hay series nuevas para procesar. Terminando.")
        return

    # Configure partial function
    worker_func = partial(
        classifier, model_other, dicom_tags_others, model_weighting, dicom_tags_weigthing, 
        model_fs, dicom_tags_fs, model_family, dicom_tags_family, model_c, model, 
        Etiquetas_dicom, Img_size, Model_name
    )

    new_results = []
    processed_count = 0
    error_count = 0

    # Parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_func, path): path for path in rutas_a_procesar}
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                        total=len(rutas_a_procesar), 
                                        desc="Procesando MRI"), 1):
            try:
                resultado_dict = future.result()
                if resultado_dict:
                    new_results.append(resultado_dict)
                    processed_count += 1
                    
                    # Update processed_series set
                    patient = resultado_dict.get('Paciente', '')
                    study = resultado_dict.get('Estudio', '')
                    serie = resultado_dict.get('Serie', '')
                    if patient and study and serie:
                        processed_series.add((str(patient), str(study), str(serie)))
                        
            except Exception as exc:
                path_fallido = futures[future]
                error_count += 1
                logging.error(f"Error processing {path_fallido}: {exc}", exc_info=True)
                
                # Add error entry
                patient, study, serie = extract_path_components(path_fallido)
                error_entry = {
                    "Paciente": patient,
                    "Estudio": study,
                    "Serie": serie,
                    "Error": f"ERROR: {str(exc)}"
                }
                new_results.append(error_entry)
            
            # Save checkpoint every 50 series
            if i % 50 == 0 and new_results:
                resultados_totales.extend(new_results)
                save_results_incrementally(resultados_totales, output_path)
                new_results = []
                logging.info(f"Checkpoint: {len(resultados_totales)} total records")

    # Final save
    if new_results:
        resultados_totales.extend(new_results)
    
    logging.info('Clasificacion acabada. Guardando CSV final...')
    print("\nGuardando resultados finales...")
    
    if resultados_totales:
        save_results_incrementally(resultados_totales, output_path, is_final=True)
        print(f"¡Clasificacion acabada! Resultados guardados en: {output_path}")
    else:
        print("No se generaron resultados.")

    # Summary
    summary = f"""
    ========== PROCESSING COMPLETE ==========
    Total records: {len(resultados_totales)}
    Newly processed: {processed_count}
    Skipped: {skipped_count}
    Errors: {error_count}
    Output: {output_path}
    Log: {log_path}
    =========================================
    """
    print(summary)
    logging.info(summary)

if __name__ == "__main__":
    main()