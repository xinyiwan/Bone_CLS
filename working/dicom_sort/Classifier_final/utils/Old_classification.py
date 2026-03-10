import pydicom
import os
import re
from utils.mr_series_classifier import *
import pandas as pd
import utils.mr_series_classifier as clas

classification_tags = ['ProtocolName',
                       'SeriesDescription',
                       'StudyDescription',
                       'SequenceName',
                       'ImageType',
                       'AcquisitionContrast',
                       'SeriesNumber'
                       ]



def get_old_classification(serie_path):
            
    try:
        class_tags = get_classification_tags(serie_path, classification_tags)
        labels, labels_from = clas.classify_serie(class_tags)
    except OSError:
        class_tags = pd.Series(index=classification_tags)
        labels = ['ERROR', 'ERROR']
        labels_from = 'UNABLE TO READ DICOM FILES'
    class_tags['labels'] = labels
    class_tags['label1'] = labels[0]
    class_tags['label2'] = np.nan
    if len(labels) > 1:
        class_tags['label2'] = labels[1]
    class_tags['labels_from'] = labels_from
    class_tags['regex'] = np.nan
    if labels_from == 'PatternMatching':
        class_tags['regex'] = clas.get_regex(class_tags)
    
    return class_tags['label1']



def get_classification_tags(serie_path, classification_tags):
    tags_serie = pd.Series(index=classification_tags)
    files = get_files_paths(serie_path)
    dcm_files_paths = [f for f in files if f.endswith('.dcm')]
    if len(dcm_files_paths) > 0:
        tags = pydicom.dcmread(dcm_files_paths[0], specific_tags=classification_tags, force=True)
        for tag in tags:
            name = tag.name.replace(' ', '').replace("'", "").replace('-', '').split('(')[0]
            if name in classification_tags:
                value = parse_str_tags_value_2(tag.value)
                tags_serie[name] = value
    return tags_serie

def get_files_paths(path):
    sub_files = [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
    return sub_files


def parse_str_tags_value_2(tag_value):
    if type(tag_value) == str:
        tag_value = tag_value.replace('-', '')
        tag_value = tag_value.replace('/', '_')
        tag_value = tag_value.replace(';', '')
        tag_value = tag_value.replace(' ', '_')
        tag_value = tag_value.replace('ä', 'a')
        tag_value = tag_value.replace(',', '')
        tag_value = tag_value.replace('.', '')
        tag_value = tag_value.replace('^', '')
    elif type(tag_value) == pydicom.multival.MultiValue:
        tag_value = '/'.join(list(tag_value))

    if not isinstance(tag_value, str):
        tag_value = str(tag_value)  # Convertir a string si no lo es   
    tag_value = re.sub(r'[^a-zA-Z0-9]', '', tag_value)

    return tag_value
