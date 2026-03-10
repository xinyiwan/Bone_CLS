from utils.dicom_tag_preprocess import *
from utils.dicom_tag_load import *
from utils.input_prepare import *
from utils.Models import *
from utils.Models_reg import *
from utils.functions_reg import *


def mr_classification( predicciones_clases_w, predicciones_clases_fs, predicciones_clases_c, predicciones_clases_f, num_vol, num_bval,bvalues):

    if predicciones_clases_w[0] == 'Other':
        name_seq = 'Other'
    elif predicciones_clases_w[0] == 'DW':
        if num_vol > 1 and num_bval > 1:
            if len(bvalues<800) > 4:
                name_seq="IVIM" 
            elif num_vol > 16* (num_bval-1)+1:
                    name_seq = "DTI"
    elif predicciones_clases_w[0] == 'T1W':
        if predicciones_clases_c==1:
            if num_vol > 1:
                if num_vol > 10:
                    name_seq = "DCE"
                else:
                    name_seq = "T1W-CESC"
            else:
                name_seq = "T1W-CE"
        else:
            if num_vol > 1:
                name_seq = "T1W-FS"
            else:
                name_seq = "T1W-FS"
    elif predicciones_clases_w[0] == 'T2W':
        if predicciones_clases_fs==1:
           name_seq = "T2W-FS"
        else:
              name_seq = "T2W"       

    return name_seq 

    """
    Function to classify MR images based on various predictions and parameters.
    
    Args:
        dicom_img (str): DICOM image file path.
        patient (str): Patient identifier.
        study (str): Study identifier.
        serie (str): Series identifier.
        predicciones_clases_w (list): Predictions for classes W.
        pred_weigthing_p (float): Weighting prediction for classes W.
        predicciones_clases_fs (list): Predictions for classes FS.
        pred_fs_p (float): Prediction for classes FS.
        predicciones_clases_c (list): Predictions for classes C.
        pred_c_p (list): Prediction probabilities for classes C.
        predicciones_clases_f (list): Predictions for scanning sequence classes.
        predicciones_clases_f_p (list): Prediction probabilities for scanning sequence classes.
        num_vol (int): Number of volumes.
        num_bval (int): Number of b-values.
        adquisition_dimension (str): Acquisition dimension.
        plano (str): Plane of the image.
        orientation (str): Orientation of the image.
        predicciones_clases_h (list): Predictions for class H.
        predicciones_clases_n (list): Predictions for class N.
        predicciones_clases_t (list): Predictions for class T.
        predicciones_clases_a (list): Predictions for class A.
        predicciones_clases_p (list): Predictions for class P.
        predicciones_clases_ll (list): Predictions for class LL.


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
        "Num Vol": num_vol,
        "Num Bval": num_bval,
        "Adquisition Dimension": adquisition_dimension,
        "Plano": plano,
        "Orientación": orientation,
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
    """