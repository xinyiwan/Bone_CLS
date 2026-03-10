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


def load_existing_results(excel_path_out):
    """Load existing results from Excel file if it exists"""
    if os.path.exists(excel_path_out):
        try:
            df_existing = pd.read_excel(excel_path_out)
            # Create a set of tuples (Paciente, Estudio, Serie) for already processed series
            processed_series = set(zip(
                df_existing['Paciente'].astype(str),
                df_existing['Estudio'].astype(str),
                df_existing['Serie'].astype(str)
            ))
            # Convert resultados list to DataFrame for return
            resultados = df_existing.to_dict('records')
            logging.info(f"Loaded {len(processed_series)} existing results from {excel_path_out}")
            return resultados, processed_series
        except Exception as e:
            logging.warning(f"Could not load existing results file: {e}")
            return [], set()
    else:
        logging.info("No existing results file found, starting fresh")
        return [], set()

def is_series_processed(patient, study, serie, processed_series):
    """Check if a series has already been processed"""
    return (str(patient), str(study), str(serie)) in processed_series

def save_results_incrementally(resultados, excel_path_out):
    """Save results to Excel file"""
    try:
        df = pd.DataFrame(resultados)
        df.to_excel(excel_path_out, index=False, engine="openpyxl")
        logging.info(f"Results saved to {excel_path_out} ({len(resultados)} total records)")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")

def create_error_result(patient, study, serie, error_type, error_message):
    """Create a standardized error result entry"""
    return {
        "Paciente": patient,
        "Estudio": study,
        "Serie": serie,
        "Nombre DICOM": "ERROR",
        "Predicción Clases W": "ERROR",
        "Predicción Clases W P": 1,
        "Predicción Clases FS": "ERROR",
        "Predicción Clases FS P": 1,
        "Predicción Clases C": "ERROR",
        "Predicción Clases C P": 1,
        "Scanning Sequence": "ERROR",
        "Scanning Sequence P": 1,
        "Num Bval": "-",
        "B-values": "-",
        "Num dim": 0,
        "Num Vol": "-",
        "Num Cortes": 0,
        "Num_dcm": "-",
        "Num_echos": 0,
        "Inversion_time": "-",
        "Adquisition Dimension": "-",
        "Plano": "-",
        "Orientación": "-",
        "Manufacturer": "Unknown",
        "ManufacturerModelName": "Unknown",
        "Fat Suppresion": "-",
        "Water Suppresion": "-",
        "In Phase": "-",
        "Out Phase": "-",
        "Real": "-",
        "Imaginary": "-",
        "Magnitude": "-",
        "Phase": "-",
        "All_image_type": "Unknown",
        "Num ImageType": 0,
        "H": "-",
        "N": "-",
        "T": "-",
        "A": "-",
        "P": "-",
        "LL": "-",
        "S": "-",
        "UL": "-",
        "Error": f"{error_type}: {error_message}"
    }

def create_skipped_result(patient, study, serie):
    """Create a result entry for skipped series (already processed)"""
    return {
        "Paciente": patient,
        "Estudio": study,
        "Serie": serie,
        "Nombre DICOM": "SKIPPED",
        "Predicción Clases W": "SKIPPED",
        "Predicción Clases W P": 1,
        "Predicción Clases FS": "SKIPPED",
        "Predicción Clases FS P": 1,
        "Predicción Clases C": "SKIPPED",
        "Predicción Clases C P": 1,
        "Scanning Sequence": "SKIPPED",
        "Scanning Sequence P": 1,
        "Num Bval": "-",
        "B-values": "-",
        "Num dim": 0,
        "Num Vol": "-",
        "Num Cortes": 0,
        "Num_dcm": "-",
        "Num_echos": 0,
        "Inversion_time": "-",
        "Adquisition Dimension": "-",
        "Plano": "-",
        "Orientación": "-",
        "Manufacturer": "Unknown",
        "ManufacturerModelName": "Unknown",
        "Fat Suppresion": "-",
        "Water Suppresion": "-",
        "In Phase": "-",
        "Out Phase": "-",
        "Real": "-",
        "Imaginary": "-",
        "Magnitude": "-",
        "Phase": "-",
        "All_image_type": "Unknown",
        "Num ImageType": 0,
        "H": "-",
        "N": "-",
        "T": "-",
        "A": "-",
        "P": "-",
        "LL": "-",
        "S": "-",
        "UL": "-",
        "Error": "SKIPPED - Already processed"
    }

def classifier(model_other,dicom_tags_others,model_weighting,dicom_tags_weigthing,model_fs,dicom_tags_fs,model_family,dicom_tags_family,model_c,model,
               path_serie,Etiquetas_dicom,Error,resultados,excel_path_out,Img_size,Model_name):

    print(path_serie)
    label_A=['2D','3D']
    label_family= ["EP","GR","IR","SE"]

    path_aux, patient = os.path.split(path_serie)
    path_aux, study = os.path.split(path_aux)
    _, serie = os.path.split(path_aux)
    e=[]
    Error=""

    try:
        
        logging.info('Starting Loading Dicom Tags')
        dicom_tags,dicom_img=cargar_dicom_tags(path_serie,Etiquetas_dicom)

        logging.info('Dicom: %s', dicom_img)

        logging.info('Starting Preprocessing Dicom Tags')
        dicom_tags_p=process_dicom_tags(dicom_tags)

        dicom_tags_p,Error=check_dicom_tags(dicom_tags_p,[])

        logging.info('Starting Prediction')

        Num_dim = obtener_valor(dicom_tags_p, 'Num_dim', 0)
        Num_img=obtener_valor(dicom_tags_p, 'Num_img', 0)
        adquisition_dimension = obtener_valor(dicom_tags_p, 'MRAcquisitionType', '-')

        # Convertir etiquetas si los valores son válidos
        try:
            adquisition_dimension = label_A[int(adquisition_dimension)] if adquisition_dimension != 'N/A' else 'Unknown'
        except Exception as e:
            logging.error(f"Error en Adquisition Dimension: {e}")
            adquisition_dimension = 'Unknown'

        if Num_dim>1 or Num_img<10:
            pred_others=False
            pred_weigthing_p=1
        else:
            try:    
                pred_others=model_other.predict(dicom_tags_p[dicom_tags_others].to_numpy()[0].reshape(1, -1) )
                pred_weigthing_p=np.max( model_other.predict_proba(dicom_tags_p[dicom_tags_others].to_numpy()[0].reshape(1, -1) ))
            except Exception as e:
                logging.error(f"Error en Scanning Sequence Family: {e}")
                pred_others = None
                pred_weigthing_p=1

        if pred_others==True:
        #### REGION CLASSIFIER  #######
            try:

                logging.info('Starting Loading Dicom Tags')
                img=load_img_C(path_serie,300)

                predicciones = model.predict(img)
                predicciones_clases_h= np.round(predicciones[0])
                predicciones_clases_n= np.round(predicciones[1])
                predicciones_clases_t= np.round(predicciones[2])
                predicciones_clases_a= np.round(predicciones[3])
                predicciones_clases_p= np.round(predicciones[4])
                predicciones_clases_ll= np.round(predicciones[5])
                predicciones_clases_s= np.round(predicciones[6])
                predicciones_clases_ul= np.round(predicciones[7])

            except Exception as e:
                if Num_dim>1: 
                    logging.info("Survey or Localizer")
                    predicciones_clases_h= [["-"]]
                    predicciones_clases_n= [["-"]]
                    predicciones_clases_t= [["-"]]
                    predicciones_clases_a= [["-"]]
                    predicciones_clases_p= [["-"]]
                    predicciones_clases_ll= [["-"]]
                    predicciones_clases_s= [["-"]]
                    predicciones_clases_ul= [["-"]]
                    Error.append("Survey or Localizer")
                else:
                    predicciones_clases_h= [[None]]
                    predicciones_clases_n= [[None]]
                    predicciones_clases_t= [[None]]
                    predicciones_clases_a= [[None]]
                    predicciones_clases_p= [[None]]
                    predicciones_clases_ll= [[None]]
                    predicciones_clases_s= [[None]]
                    predicciones_clases_ul= [[None]]
                    Error.append(e)

            try:
                pred_weigthing = model_weighting.predict(dicom_tags_p[dicom_tags_weigthing].to_numpy()[0].reshape(1, -1))
                pred_weigthing_p =np.max( model_weighting.predict_proba(dicom_tags_p[dicom_tags_weigthing].to_numpy()[0].reshape(1, -1)))
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                pred_weigthing = [[None]]  # Valores por defecto

            predicciones_clases_w = pred_weigthing[0]
            predicciones_clases_c = "-"
            pred_c_p = [1]
            predicciones_clases_fs = "-"
            pred_fs_p = 1
            predicciones_clases_f_p=1

            if pred_weigthing[0]=='T2W' or pred_weigthing[0]=='T1W':
                
                try:
                    pred_fs = model_fs.predict(dicom_tags_p[dicom_tags_fs].to_numpy()[0].reshape(1, -1))
                    pred_fs_p =np.max(model_fs.predict_proba(dicom_tags_p[dicom_tags_fs].to_numpy()[0].reshape(1, -1)))
                    predicciones_clases_fs = 'Y' if pred_fs[0]==True else 'N'    

                except Exception as e:
                    logging.error(f"Error in prediction: {e}")
                    #pred_weigthing = [None, None, None]  # Valores por defecto
                    predicciones_clases_fs = None  # Valores por defecto

                
                logging.info('Starting Loading input')

                if pred_weigthing[0]=='T1W':

                    img=cargar_entrada_img(dicom_img,Img_size,Model_name)

                    try:
                        pred_c = model_c.predict([img])
                        pred_c_p =pred_c[0] if np.round(pred_c[0])==1 else 1-pred_c[0]   
                        predicciones_clases_c = 'Y' if np.round(pred_c[0])==1 else 'N'    

                    except Exception as e:
                        logging.error(f"Error in prediction: {e}")
                        #pred_weigthing = [None, None, None]  # Valores por defecto
                        predicciones_clases_c = None   # Valores por defecto
                
                logging.info('Prediction Done')

            try:
                predicciones_clases_f = scanning_sequence(dicom_tags)
            except Exception as e:
                logging.error(f"Error en Scanning Sequence: {e}")
                predicciones_clases_f = 'Unknown'
                predicciones_clases_f_p=1

            if predicciones_clases_f=="Unknown" or predicciones_clases_f=="RM":
                try:
                    predicciones_clases_f=model_family.predict(dicom_tags_p[dicom_tags_family].to_numpy()[0].reshape(1, -1) )
                    predicciones_clases_f_p =np.max(model_family.predict_proba(dicom_tags_p[dicom_tags_family].to_numpy()[0].reshape(1, -1) ))
                    predicciones_clases_f=label_family[np.squeeze(predicciones_clases_f)]

                except Exception as e:
                    logging.error(f"Error en Scanning Sequence Family: {e}")
                    predicciones_clases_f = 'Unknown'       


        elif pred_others==False or pred_others==None:
            if Num_dim>1 or Num_img<10: 
                logging.info("Survey or Localizer")
                predicciones_clases_h= [["-"]]
                predicciones_clases_n= [["-"]]
                predicciones_clases_t= [["-"]]
                predicciones_clases_a= [["-"]]
                predicciones_clases_p= [["-"]]
                predicciones_clases_ll= [["-"]]
                predicciones_clases_s= [["-"]]
                predicciones_clases_ul= [["-"]]
                Error.append("Survey or Localizer")
            else:
                predicciones_clases_h= [[None]]
                predicciones_clases_n= [[None]]
                predicciones_clases_t= [[None]]
                predicciones_clases_a= [[None]]
                predicciones_clases_p= [[None]]
                predicciones_clases_ll= [[None]]
                predicciones_clases_s= [[None]]
                predicciones_clases_ul= [[None]]

            pred_c_p = [1]
            pred_fs_p = 1
            predicciones_clases_f_p=1
            Error=[]

            if pred_others==False or dicom_img=="Error":
                predicciones_clases_w = ['Other']
                predicciones_clases_fs = '-'    
                predicciones_clases_c = '-'
                predicciones_clases_f = '-'
            else:
                predicciones_clases_w = ["Error"]
                predicciones_clases_fs =  None   
                predicciones_clases_c =  None
                predicciones_clases_f =  None
                pred_weigthing_p= 1


        logging.info('Saving Prediction')

        # Almacenar resultados de forma segura
    except Exception as e:
        
        error_msg = f"Error en Paciente: {patient}, Estudio: {study}, Serie: {serie}, DICOM: {dicom_img if 'dicom_img' in locals() else None}\n{traceback.format_exc()}"
        logging.error(error_msg)
        pred_weigthing_p= 1
        pred_c_p = [1]
        pred_fs_p = 1
        predicciones_clases_f_p=1
        predicciones_clases_h= [[None]]
        predicciones_clases_n= [[None]]
        predicciones_clases_t= [[None]]
        predicciones_clases_a= [[None]]
        predicciones_clases_p= [[None]]
        predicciones_clases_ll= [[None]]
        predicciones_clases_s= [[None]]
        predicciones_clases_ul= [[None]]
        
        try:
            if dicom_img=="Error":
                predicciones_clases_w=['Other']
                predicciones_clases_fs = '-'
                predicciones_clases_c = '-'
                predicciones_clases_f = '-'
            else:
                predicciones_clases_w=["Error"]
                predicciones_clases_fs = 'Error'
                predicciones_clases_c = 'Error'
                predicciones_clases_f = 'Error'
        except Exception as e:
                predicciones_clases_w=["Error"]
                predicciones_clases_fs = 'Error'
                predicciones_clases_c = 'Error'
                predicciones_clases_f = 'Error'
    

    resultados.append({
        "Paciente": patient,
        "Estudio": study,
        "Serie": serie,
        "Nombre DICOM": dicom_img,  
        "Predicción Clases W": predicciones_clases_w[0],
        "Predicción Clases W P": pred_weigthing_p,
        "Predicción Clases FS": predicciones_clases_fs,  
        "Predicción Clases FS P": pred_fs_p,  
        "Predicción Clases C": predicciones_clases_c,  
        "Predicción Clases C P": pred_c_p[0],  
        "Scanning Sequence": predicciones_clases_f,
        "Scanning Sequence P": predicciones_clases_f_p,
        "Num Bval": obtener_valor(dicom_tags_p, 'b_value_num', '-'),
        "B-values": obtener_valor(dicom_tags_p, 'bvalues', '-'),
        "Num dim": Num_dim,        
        "Num Vol": obtener_valor(dicom_tags_p, 'Num_vol', '-'),
        "Num Cortes": Num_img ,
        "Num_dcm": obtener_valor(dicom_tags_p, 'Num_dcm', '-'),
        "Num_echos":obtener_valor(dicom_tags_p, 'EchoNumbers', 0),
        "Inversion_time":obtener_valor(dicom_tags_p, 'InversionTime', '-'),
        "Adquisition Dimension": obtener_valor(dicom_tags_p, 'MRAcquisitionType', '-'),
        "Plano": obtener_valor(dicom_tags_p, 'Plano', '-'),
        "Orientación": obtener_valor(dicom_tags_p, 'Orientacion', '-'),
        "Manufacturer": obtener_valor(dicom_tags_p, 'Manufacturer', 'Unknown'),
        "ManufacturerModelName": obtener_valor(dicom_tags_p, 'ManufacturerModelName', 'Unknown'),
        "Fat Suppresion": obtener_valor(dicom_tags_p, 'F', '-'),
        "Water Suppresion": obtener_valor(dicom_tags_p, 'W', '-'),
        "In Phase": obtener_valor(dicom_tags_p, 'IP', '-'),
        "Out Phase": obtener_valor(dicom_tags_p, 'OP', '-'),
        "Real": obtener_valor(dicom_tags_p, 'ImageType_R', '-'),
        "Imaginary": obtener_valor(dicom_tags_p, 'ImageType_I', '-'),
        "Magnitude": obtener_valor(dicom_tags_p, 'ImageType_M', '-'),
        "Phase": obtener_valor(dicom_tags_p, 'ImageType_P', '-'),
        "All_image_type": obtener_valor(dicom_tags, 'All_image_type', 'Unknown'),
        "Num ImageType": len(obtener_valor(dicom_tags, 'All_image_type', 'Unknown')),
        "H": predicciones_clases_h[0][0],
        "N": predicciones_clases_n[0][0],
        "T": predicciones_clases_t[0][0],
        "A":predicciones_clases_a[0][0],
        "P": predicciones_clases_p[0][0],
        "LL": predicciones_clases_ll[0][0],
        "S": predicciones_clases_s[0][0],
        "UL": predicciones_clases_ul[0][0],
        "Error": Error
        }) 
    
    df = pd.DataFrame(resultados)
    df.to_excel(excel_path_out, index=False, engine="openpyxl")

    return resultados


def create_error_result(patient, study, serie, error_type, error_message):
    """Create a standardized error result entry"""
    return {
        "Paciente": patient,
        "Estudio": study,
        "Serie": serie,
        "Nombre DICOM": "ERROR",
        "Predicción Clases W": "ERROR",
        "Predicción Clases W P": 1,
        "Predicción Clases FS": "ERROR",
        "Predicción Clases FS P": 1,
        "Predicción Clases C": "ERROR",
        "Predicción Clases C P": 1,
        "Scanning Sequence": "ERROR",
        "Scanning Sequence P": 1,
        "Num Bval": "-",
        "B-values": "-",
        "Num dim": 0,
        "Num Vol": "-",
        "Num Cortes": 0,
        "Num_dcm": "-",
        "Num_echos": 0,
        "Inversion_time": "-",
        "Adquisition Dimension": "-",
        "Plano": "-",
        "Orientación": "-",
        "Manufacturer": "Unknown",
        "ManufacturerModelName": "Unknown",
        "Fat Suppresion": "-",
        "Water Suppresion": "-",
        "In Phase": "-",
        "Out Phase": "-",
        "Real": "-",
        "Imaginary": "-",
        "Magnitude": "-",
        "Phase": "-",
        "All_image_type": "Unknown",
        "Num ImageType": 0,
        "H": "-",
        "N": "-",
        "T": "-",
        "A": "-",
        "P": "-",
        "LL": "-",
        "S": "-",
        "UL": "-",
        "Error": f"{error_type}: {error_message}"
    }

def main():


    parser = argparse.ArgumentParser(
        description="""
        Código para clasificar las secuencias segun su ponderación, supresion grasa, contraste y familia; separando los casos que no son usados para analisis (Mapas, localizadores...) 
        a partir de un excel de entrada con las rutas a las series que se quieren clasificar o una carpeta con los pacientes que se quieren clasificar. El resultado es un excel con las 
        rutas y algunas de las caracteristicas de cada imagen.

        Ejemplo de ejecución:
        docker run --rm --env-file=.env -v C:\path\of\Project:/Proyecto  -v C:\path\of\Parameterfolder:/Parameters_config  -v C:\path\of\Path_excel:/Path_excel -t dockername:tag 

        donde:
        * C:\path\of\Project ruta del proyecto
        * C:\path\of\Parameterfolder ruta de la carpeta donde se encuentra el fichero de configuración de parámetros
        * C:\Path_excel (opcional) ruta a la carpeta donde se encuentra el excel con las rutas de las series a clasificar""",

        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args = parser.parse_args()

    Etiquetas_dicom =['EchoTime', 'InversionTime',  'PixelSpacing', 'RepetitionTime','SliceThickness','MRAcquisitionType','FlipAngle',
                   'PixelBandwidth','ImageType','SequenceVariant','ScanningSequence','ImageOrientationPatient','EchoTrainLength',
                   'ScanOptions','SpectrallySelectedSuppression','Manufacturer','ManufacturerModelName']



    path_project = os.environ.get('INPUT_PATH',"/Proyecto")
    config = os.environ.get('CONFIG_PATH',"/Parameters_config")
    config_filename = os.environ.get('CONFIG_FILE', 'parameter_configuration.json')
    config_file = os.path.join(config, config_filename)

    os.makedirs(os.path.join(path_project,'Results'),exist_ok=True)
    os.makedirs(os.path.join(path_project,'Logs'),exist_ok=True)

    excel_path_out=os.path.join(path_project,'Results','Sequence_Classifier_2.xlsx')
    log_path=os.path.join(path_project,'Logs','Sequence_Classifier.log')


    logging.basicConfig(
        filename=log_path,  # Nombre del archivo de registro
        level=logging.INFO,       # Nivel de registro (puedes usar logging.DEBUG, logging.INFO, logging.WARNING, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info('Starting Sequence Classifier')

    logging.info('Loading Parameters file')

    with open(config_file, 'r') as f:
        params_config = json.load(f)

    data = params_config.get('CLASSIFIER', {})


    path_model_folder_config="/module"


    logging.info('Loading Other Model Configuration')

    # OTHER MODEL
    path_model_other=os.path.join(path_model_folder_config,'Models/Model_others.pkl')
    path_dicom_tags_others=os.path.join(path_model_folder_config,"Models/Model_others.json")
    with open(path_model_other, "rb") as archivo:
        model_other = pickle.load(archivo)
    with open(path_dicom_tags_others, 'r') as f:
        data_o = json.load(f)
        dicom_tags_others=data_o['ETIQUETAS_DICOM']

    logging.info('Loading Weigthing Model Configuration')

    # WEIGTHING MODEL
    path_model_weigthing=os.path.join(path_model_folder_config,'Models/Model_weigthing.pkl')
    path_dicom_tags_weigthing=os.path.join(path_model_folder_config,"Models/Model_weigthing.json")
    with open(path_model_weigthing, "rb") as archivo:
        model_weighting = pickle.load(archivo)
    with open(path_dicom_tags_weigthing, 'r') as f:
        data_w = json.load(f)
        dicom_tags_weigthing=data_w['ETIQUETAS_DICOM']

    logging.info('Loading Family Model Configuration')

    # FAMILY MODEL
    path_model_family=os.path.join(path_model_folder_config,'Models/Model_family.pkl')
    path_dicom_tags_family=os.path.join(path_model_folder_config,"Models/Model_family.json")
    with open(path_model_family, "rb") as archivo:
        model_family = pickle.load(archivo)
    with open(path_dicom_tags_family, 'r') as f:
        data_f = json.load(f)
        dicom_tags_family=data_f['ETIQUETAS_DICOM']

    logging.info('Loading FS Model Configuration')

    # FAT SUPRESSION MODEL
    path_model_fs=os.path.join(path_model_folder_config,'Models/Model_fs.pkl')
    path_dicom_tags_fs=os.path.join(path_model_folder_config,"Models/Model_fs.json")
    with open(path_model_fs, "rb") as archivo:
        model_fs = pickle.load(archivo)
    with open(path_dicom_tags_fs, 'r') as f:
        data_fs = json.load(f)
        dicom_tags_fs=data_fs['ETIQUETAS_DICOM']

    logging.info('Loading Contrast Model Configuration')

    # CONTRAST MODEL
    path_data_c=os.path.join(path_model_folder_config,"Models/Model_c.json")
    with open(path_data_c, 'r') as f:
        data_model_c = json.load(f)
    path_model_c=os.path.join(path_model_folder_config,'Models/Model_c.h5')


    Model_type = data_model_c['MODEL']
    Training_mode=data_model_c['TRAINING_MODE']
    learning_rate = data_model_c['LEARNING_RATE']
    Model_name = data_model_c['MODEL_NAME']
    Img_size=data_model_c['IMG_SIZE']

    model_c = crear_modelo_img(Model_type,Training_mode,["1"],learning_rate,'c',Img_size,'sigmoid','binary_crossentropy',Model_name)   
    model_c.load_weights(path_model_c)

    logging.info('Loading Model Configuration')



    # REGION MODEL
    path_dicom_tags=os.path.join(path_model_folder_config,"Models/Model_region.json")
    with open(path_dicom_tags, 'r') as f:
        data_model_reg = json.load(f)
    path_model=os.path.join(path_model_folder_config,'Models/Model_region.h5')



    Model_type_reg = data_model_reg['MODEL']
    Labels = data_model_reg['LABELS']
    Model_name_reg = data_model_reg['MODEL_NAME']
    Img_size_reg=data_model_reg['IMG_SIZE']

    model = crear_modelo_reg(Model_type_reg,Labels,Img_size_reg,Model_name_reg)   
    model.load_weights(path_model)

    study = ''
    patient = ''
    serie = ''
    resultados = []
    Error=""

    logging.info('Loading Model')

    logging.info('Starting Prediction')

    excel_flag=data.get('excel_flag', False)
    if excel_flag:
        excel_path = data.get('excel_path', None)
        excel_column= data.get('excel_column', None)

        if excel_path is None or excel_column is None:
            raise ValueError("Faltan 'excel_path' o 'excel_column' en la configuración.")

        # Cargar Excel
        df_datos = pd.read_excel(excel_path)

        for _, row in df_datos.iterrows():
            path_serie = row[excel_column]
            resultados = classifier(model_other,dicom_tags_others,model_weighting,dicom_tags_weigthing,model_fs,dicom_tags_fs,model_family,dicom_tags_family,model_c,model,
                                                  path_serie,Etiquetas_dicom,Error,resultados,excel_path_out,Img_size,Model_name)

    else: 
        path_adquisition=os.path.join(path_project, data.get('input_folder', 'ADQUISICIONES'))
        patients=os.listdir(path_adquisition)
        
    for patient in tqdm(patients):
        print(patient)
        path_patient = os.path.join(path_adquisition, patient)
        
        if os.path.isdir(path_patient):
            try:
                studies = os.listdir(path_patient)
            except Exception as e:
                error_msg = f"Error accessing patient directory {path_patient}: {str(e)}"
                logging.error(error_msg)
                print(f"ERROR: {error_msg}")
                continue  # Skip to next patient
                
            for study in studies:
                path_study = os.path.join(path_patient, study)
                
                if not os.path.isdir(path_study):
                    continue
                    
                try:
                    series = os.listdir(path_study)
                except Exception as e:
                    error_msg = f"Error accessing study directory {path_study}: {str(e)}"
                    logging.error(error_msg)
                    print(f"ERROR: {error_msg}")
                    continue  # Skip to next study
                    
                if 'MR' in study:
                    for serie in series:
                        path_serie = os.path.join(path_adquisition, patient, study, serie)
                        
                        try:
                            # Validate path exists before processing
                            if not os.path.exists(path_serie):
                                raise FileNotFoundError(f"Path does not exist: {path_serie}")
                                
                            resultados = classifier(
                                model_other, dicom_tags_others,
                                model_weighting, dicom_tags_weigthing,
                                model_fs, dicom_tags_fs,
                                model_family, dicom_tags_family,
                                model_c, model,
                                path_serie, Etiquetas_dicom, Error, 
                                resultados, excel_path_out, Img_size, Model_name
                            )
                            logging.info('Serie processed: %s, %s, %s', patient, study, serie)
                            
                        except FileNotFoundError as e:
                            error_msg = f"File not found - Patient: {patient}, Study: {study}, Serie: {serie}, Error: {str(e)}"
                            logging.error(error_msg)
                            print(f"ERROR: {error_msg}")
                            
                            # Add error entry to results
                            resultados.append(create_error_result(
                                patient, study, serie, "FileNotFoundError", str(e)
                            ))
                            
                        except Exception as e:
                            error_msg = f"Unexpected error - Patient: {patient}, Study: {study}, Serie: {serie}, Error: {str(e)}"
                            logging.error(error_msg)
                            logging.error(traceback.format_exc())
                            print(f"ERROR: {error_msg}")
                            
                            # Add error entry to results
                            resultados.append(create_error_result(
                                patient, study, serie, "UnexpectedError", str(e)
                            ))
                        
                        # Save results after each serie to prevent data loss
                        try:
                            df = pd.DataFrame(resultados)
                            df.to_excel(excel_path_out, index=False, engine="openpyxl")
                        except Exception as e:
                            logging.error(f"Error saving results after serie {serie}: {str(e)}")

    logging.info('Clasificacion acabada')
    print("Clasificacion acabada")

    logging.shutdown()

if __name__ == "__main__":
    main()