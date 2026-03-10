import re
import numpy as np


def regex_search_label(text, regexes):
    if any(regex.search(text) for regex in regexes):
        return True
    else:
        return False



adc = [re.compile('_ADC', re.IGNORECASE),
       re.compile('_TRACEW$', re.IGNORECASE),
       re.compile('_ColFA$', re.IGNORECASE),
       re.compile('_FA$', re.IGNORECASE),
       re.compile('_EXP$', re.IGNORECASE),
       re.compile('Apparent[-_]?Diffusion[-_]?Coefficient', re.IGNORECASE)
       ]

proc = [re.compile('proces', re.IGNORECASE),
        re.compile('projection', re.IGNORECASE),
        re.compile('ADC', re.IGNORECASE),
        re.compile('EXP', re.IGNORECASE),
        re.compile('Apparent[ -_]?Diffusion[ -_]?Coefficient', re.IGNORECASE),
        re.compile('(?=.*screen)(?=.*save)', re.IGNORECASE),
        re.compile('.*screenshot', re.IGNORECASE),
        re.compile('.*screensave', re.IGNORECASE),
        re.compile('sub', re.IGNORECASE)
        ]

diffusion = [re.compile('dif', re.IGNORECASE),
             re.compile('dw', re.IGNORECASE),
             #re.compile('(?=.*diff)(?=.*dir)', re.IGNORECASE),
             re.compile('hardi', re.IGNORECASE),
             re.compile('ep_b[0-9]'),
             re.compile('re_b[0-9]'),
             re.compile('[i]?s[o]?b[ _-]?[0-9]', re.IGNORECASE)
             ]
rsfMRI= [re.compile('rsfMRI', re.IGNORECASE)]

SWI= [re.compile('SWI', re.IGNORECASE)]

qsm= [re.compile('QSM', re.IGNORECASE)]


dti=[re.compile('dti', re.IGNORECASE)]

perfusion = [re.compile('lava', re.IGNORECASE),
             re.compile('vibe', re.IGNORECASE),
             #re.compile('in_fs', re.IGNORECASE),
             re.compile('thrive', re.IGNORECASE),
             #re.compile('water_', re.IGNORECASE),
             re.compile('fame', re.IGNORECASE),
             #re.compile('asl', re.IGNORECASE),
             #re.compile('(?=.*blood)(?=.*flow)', re.IGNORECASE),
             #re.compile('(?=.*art)(?=.*spin)', re.IGNORECASE),
             #re.compile('tof', re.IGNORECASE),
             re.compile('perfusion', re.IGNORECASE)]

stir = [re.compile('s[ _-]?tir', re.IGNORECASE)]

in_out = [re.compile('in[-_]?out', re.IGNORECASE),
          re.compile('(ax(i|ial)?|cor(onal)?|sag(ital)?)[ -_]?(in|out)', re.IGNORECASE),
          re.compile('(fase)+.*(fase)+', re.IGNORECASE),
          re.compile('enfase', re.IGNORECASE),
          re.compile('FASE/(F|OPU)', re.IGNORECASE),
          re.compile('in[ -_]?phase|out([ -_]|of)?phase', re.IGNORECASE)
          ]


screen = [re.compile('(?=.*screen)(?=.*save)', re.IGNORECASE),
          re.compile('.*screenshot', re.IGNORECASE),
          re.compile('.*screensave', re.IGNORECASE)
          ]

cal = [re.compile('(?=.*asset)(?=.*cal)', re.IGNORECASE),
       #re.compile('^asset$', re.IGNORECASE),
       re.compile('calib', re.IGNORECASE)
       ]

loc = [re.compile('localizer', re.IGNORECASE),
       re.compile('localiser', re.IGNORECASE),
       re.compile('surv', re.IGNORECASE),
       re.compile('survey', re.IGNORECASE),
       re.compile('scout', re.IGNORECASE),
       re.compile('topogram', re.IGNORECASE),
       re.compile('loc', re.IGNORECASE),
       #re.compile(r'\bscout\b', re.IGNORECASE),
       re.compile('(?=.*plane)(?=.*loc)', re.IGNORECASE),
       #re.compile('(?=.*plane)(?=.*survey)', re.IGNORECASE),
       re.compile('3-plane', re.IGNORECASE),
       #re.compile('^loc*', re.IGNORECASE),
       re.compile('Scout', re.IGNORECASE),
       re.compile('AdjGre', re.IGNORECASE)
       ]

flair = [re.compile('flair', re.IGNORECASE)]

pd = [#re.compile('^PD$'),
      re.compile('(?=.*proton)(?=.*density)', re.IGNORECASE),
      re.compile('pd_'),
      re.compile('_pd')
      ]

t1 = [re.compile('t1', re.IGNORECASE),
      #re.compile('t1w', re.IGNORECASE),
      #re.compile('(?=.*3d anat)(?![inplane])', re.IGNORECASE),
      #re.compile('(?=.*3d)(?=.*bravo)(?![inplane])', re.IGNORECASE),
      re.compile('spgr', re.IGNORECASE),
      re.compile('tfl', re.IGNORECASE),
      re.compile('mprage', re.IGNORECASE),
      re.compile('(?=.*mm)(?=.*iso)', re.IGNORECASE),
      re.compile('(?=.*mp)(?=.*rage)', re.IGNORECASE)
      ]


t2 = [re.compile('t2', re.IGNORECASE),
      re.compile('fiesta', re.IGNORECASE),
      re.compile('ssfse', re.IGNORECASE),
      re.compile('frfse', re.IGNORECASE)
      ]

ivim = [re.compile('IVIM', re.IGNORECASE)
      ]

dixon = [re.compile('mDIXON', re.IGNORECASE)
      ]

swan = [re.compile('swan', re.IGNORECASE),
        re.compile('SWI', re.IGNORECASE)
      ]

mtonoff = [re.compile('MT_OFF_ON', re.IGNORECASE),
          re.compile('MTONOFF', re.IGNORECASE)
      ]


series_types_dict = {# 'ADC map': adc,
                     'PROCESSED': proc,
                     'DW': ivim,
                     'DW': diffusion,
                     'PW': perfusion,
                     'STIR': stir,
                     'ChS': in_out,
                     #'SCREENSAVE': screen,
                     'CALIBRATION': cal,
                     'LOCALIZER': loc,
                     'FLAIR': flair,
                     'T1W': t1,
                     'T2W': t2,
                     'PDW': pd,
                     'SWAN':swan,
                     'mDIXON':dixon,
                     'MTONOFF': mtonoff,
                     'DTI':dti,
                     'rsfMRI':rsfMRI,
                     'QSM':qsm,
                     }

def get_description_text(serie, tag):
    words_to_remove = [re.compile('pat2', re.IGNORECASE),
                       re.compile('miopatiasdifusas', re.IGNORECASE),
                       re.compile('protocol', re.IGNORECASE),
                       re.compile('maxilocuello', re.IGNORECASE)
                       ]
    
    text = []
    #for tag in ['SeriesDescription', 'ProtocolName', 'SequenceName']:
    if tag in serie.index:
        text.append(str(serie[tag]))
    text = '_'.join(text)
    
    for word in words_to_remove:
        text = re.sub(word, '', text)
    
    return text
    

def get_label_from_acq_contrast(serie):
    if 'AcquisitionContrast' in serie.index:
        acq = str(serie['AcquisitionContrast'])
        if acq == 'T1':
            return ['T1']
        elif acq == 'T2':
            return ['T2']
        elif acq == 'DIFFUSION':
            return ['DIFFUSION']
        elif acq == 'PROTON_DENSITY':
            return 'PROTON_DENSITY'
        else:
            return ['UNKNOWN']
    else:
        return ['UNKNOWN']


def classify_serie(serie):
    
    #text = get_description_text(serie)
    
    labels = []
    labels_from = ''
    text = get_description_text(serie, 'SeriesDescription')
    for label, regexes in series_types_dict.items():
        if regex_search_label(text, regexes):
            #print('Found in SD: ', label)
            labels.append(label)
            labels_from = 'PatternMatching_SeriesDescription'
    #print(len(labels))
    if len(labels)==0:
        text = get_description_text(serie, 'SequenceName')
        for label, regexes in series_types_dict.items():
            if regex_search_label(text, regexes):
                #print('Found in SN: ', label)
                labels.append(label)
                labels_from = 'PatternMatching_SequenceName'
    #print(len(labels))
    if len(labels)==0:
        text = get_description_text(serie, 'ProtocolName')
        for label, regexes in series_types_dict.items():
            if regex_search_label(text, regexes):
                #print('Found in PN: ', labels)
                labels.append(label)
                labels_from = 'PatternMatching_ProtocolName'
    #print(len(labels))
    
    if 'LOCALIZER' in str(serie['ImageType']):
        labels = ['LOCALIZER']
        labels_from = 'ImageType'
    elif 'ADC' in str(serie['ImageType']) or 'DERIVED/SECONDARY/' in str(serie['ImageType']):
        labels = ['PROCESSED']
        labels_from = 'ImageType'
    elif re.search((re.compile('(?=.*screen)(?=.*save)', re.IGNORECASE)), str(serie['ImageType'])):
        labels = ['PROCESSED']
        labels_from = 'ImageType'
    elif len(labels) == 0:
        labels = get_label_from_acq_contrast(serie)
        labels_from = 'AcquisitionContrast'
    
    if labels == ['UNKNOWN']:
        labels_from = np.nan
    
    return labels, labels_from


def get_regex(serie):
    text = get_description_text(serie)
    regex_list = []
    for _, regexes in series_types_dict.items():
        for regex in regexes:
            if regex.search(text):
                regex_list.append(regex)
    return regex_list


def test_regex(df_series):
    regex_dict = {}
    for label, regexes in series_types_dict.items():
        regex_dict[label] = {}
        for reg in regexes:
            regex_dict[label][reg] = 0
        for i in df_series['churro']:
            for reg in regexes:
                if reg.search(i):
                    regex_dict[label][reg] += 1
    return regex_dict
