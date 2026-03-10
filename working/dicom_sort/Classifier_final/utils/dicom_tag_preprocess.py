import pandas as pd
import logging

def map_column_values(df, columns_to_map, mapping_dict):

    # Iterar sobre las columnas especificadas y aplicar el mapeo
    for col in columns_to_map:
        if col in df.columns:
            df[col] = df[col].map(mapping_dict[col]).fillna(df[col])
    return df


def custom_one_hot_encoding(df, columns, specific_values=None):

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        
        # Obtener los valores únicos en la columna
        unique_values = df[col].dropna().unique()
        
        # Usar valores específicos si están definidos
        if specific_values is not None:
            values_to_encode = [val for val in specific_values if val in unique_values]
        else:
            values_to_encode = unique_values
        
        # Crear columnas one-hot para los valores seleccionados
        for value in values_to_encode:
            col_name = f"{value}"
            df[col_name] = (df[col] == value).astype(int)
    
    return df

def one_hot_encoding_by_prefix(df, prefixes, possible_values_ori):
    
    new_columns = {}  # Almacenar las nuevas columnas one-hot encoding
    df = df.copy()  # Copia para no modificar el original

    for prefix in prefixes:
        # Filtrar las columnas que comienzan con el prefijo
        columns_to_process = [col for col in df.columns if col.startswith(prefix)]
        if prefix not in possible_values_ori.keys():
            continue  # Saltar si el prefijo no tiene valores mapeados en el diccionario
        
        # Obtener los valores posibles para este prefijo
        possible_values = possible_values_ori[prefix]
        
        # Crear columnas one-hot para los valores especificados
        for value in possible_values:
            col_name = f"{prefix}_{value}"

            def cell_contains_value(cell):
                if pd.isna(cell):
                    return 0
                if isinstance(cell, str):
                    parts = [p.strip() for p in cell.split("'")]
                    # Nos quedamos con las entradas válidas (no vacías ni separadores como , o [ o ])
                    return int(any(elem in value for elem in parts))
                return 0

            # Aplica la función a cada celda de las columnas del prefijo y usa any por fila
            new_columns[col_name] = df[columns_to_process].applymap(cell_contains_value).any(axis=1).astype(int)
        # Eliminar las columnas originales
        df.drop(columns=columns_to_process, inplace=True)
    
    # Añadir las nuevas columnas al DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    return df

def one_hot_encoding_by_prefix_ori(df, prefixes, possible_values_ori):
    
    new_columns = {}  # Almacenar las nuevas columnas one-hot encoding
    df = df.copy()  # Copia para no modificar el original

    for prefix in prefixes:
        # Filtrar las columnas que comienzan con el prefijo
        columns_to_process = [col for col in df.columns if col.startswith(prefix)]
        if prefix not in possible_values_ori.keys():
            continue  # Saltar si el prefijo no tiene valores mapeados en el diccionario
        
        # Obtener los valores posibles para este prefijo
        possible_values = possible_values_ori[prefix]
        
        # Crear columnas one-hot para los valores especificados
        for value in possible_values:
            if prefix.startswith('Old'):
                col_name = f"{value}"
            else:
                col_name = f"{prefix}_{value}"
            new_columns[col_name] = (
                df[columns_to_process].apply(lambda row: row.isin([value]).any(), axis=1).astype(int)
            )
        
        # Eliminar las columnas originales
        df.drop(columns=columns_to_process, inplace=True)   

    # Añadir las nuevas columnas al DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    return df

"""
def or_columns(df,mapping_or,delete_col):

    for feature in mapping_or:

        nueva_columna=feature
        col1=sorted(mapping_or[feature])[0]
        col2=sorted(mapping_or[feature])[1]
        # Aplicar la operación lógica OR
        df[nueva_columna] = (df[col1] | df[col2]).astype(int)
        if delete_col:
            df = df.drop(columns=[col1, col2])

    return df
"""

def or_columns(df, mapping_or, delete_col):
    for feature, cols in mapping_or.items():
        existing_cols = [col for col in cols if col in df.columns]  # Filtrar columnas existentes
        
        if len(existing_cols) > 0:  # Verificar que haya al menos una columna en el DataFrame
            df[feature] = df[existing_cols].any(axis=1).astype(int)  # OR de todas las columnas existentes
            
            if delete_col:
                df.drop(columns=existing_cols, inplace=True)  # Eliminar las columnas originales
    
    return df

# ============================
# CONFIGURACIÓN DEL PROGRAMA
# ============================
columns_to_map=['ImageType_1','ImageType_2','MRAcquisitionType','PhotometricInterpretation','SpectrallySelectedSuppression']
columns_to_ohc=['Old_Class']
columns_to_ohc_pre_ori=['SequenceVariant','ScanningSequence','ScanOptions','Old_Class']
columns_to_ohc_pre=['ImageType']

mapping_dict_map={'ImageType_1':{              
                'ORIGINAL':1,
                'DERIVED':0},
              'ImageType_2':{
                'PRIMARY':1,
                'SECONDARY':0},

              'MRAcquisitionType':{
                '2D':0,
                '3D':1,},
              'SpectrallySelectedSuppression':{
                'NONE':0,
                'WATER':0,
                'FAT_AND_WATER':1,
                'SILICON_GEL':0,
                'FAT':1,},
              'Plano':{
                'Axial':0,
                'Sagital':1,
                'Coronal':2
                },
            'Orientacion':{
                'RA':0,
                'AF':1,
                'RAF':2,
                'RAH':3,
                'RF':4                
                },
              }


mapping_dict_ohc_pre_ori={'SequenceVariant':{'OSP','SK','SS','MP','MTC','SP'},
                      'ScanningSequence':{'EP','GR','IR','MR','SE'},
                      'ScanOptions':{'PER','RG','CG','PPG','FC','PFF','PFP','SP','FS'},
                      'Old_Class':{'T2W','T1W','LOCALIZER','DW','FLAIR','PW','SWAN','PDW','ChS','STIR','mDIXON','UNKNOWN','PROCESSED'}
                        }   												
mapping_dict_ohc_pre={'ImageType':{'WATER','W','FAT','F','IN_PHASE','IP','OPP_PHASE','OUT_PHASE','OP','ADC', 'OTHER',"I","R","P","M"}
                        }   

mapping_or_image_type_1={'W':{'ImageType_WATER','ImageType_W'},
            'F':{'ImageType_FAT','ImageType_F'},
            'IP':{'ImageType_IN_PHASE','ImageType_IP'},
            'OP':{'ImageType_OUT_PHASE','ImageType_OP','ImageType_OPP_PHASE'}
                        }  
mapping_or_image_type_2={'DIXON_W':{'W','F','IP','OP'}}   


def process_dicom_tags(df):

    df=map_column_values(df, columns_to_map, mapping_dict_map)
    #df=custom_one_hot_encoding(df, columns_to_ohc)
    df=one_hot_encoding_by_prefix(df, columns_to_ohc_pre,mapping_dict_ohc_pre)
    df=one_hot_encoding_by_prefix_ori(df, columns_to_ohc_pre_ori,mapping_dict_ohc_pre_ori)   
    df = or_columns(df, mapping_or_image_type_1,False)
    df = or_columns(df, mapping_or_image_type_2,False)

    return df


def scanning_sequence(dicom_tags):

    try:
        tag=dicom_tags["ScanningSequence"].values[0]
    except:
        tag_1=dicom_tags["ScanningSequence_1"].values[0]
        tag_2=dicom_tags["ScanningSequence_2"].values[0]
        tag=tag_1+','+tag_2

    return tag


def obtener_valor(dic, clave, default='Desconocido'):
    try:
        return dic.get(clave, pd.Series([default])).values[0]
    except Exception as e:
        logging.error(f"Error obteniendo {clave}: {e}")
        return default

