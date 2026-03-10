from tensorflow.keras.applications import ResNet50,EfficientNetB3,ConvNeXtBase
from tensorflow.keras.layers import  Dense, Input, GlobalAveragePooling2D,Conv2D,Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils.metrics import *
from tensorflow.keras.metrics import AUC,BinaryAccuracy



def crear_modelo_reg(MODEL,Labels,Img_size,name_model='EfficientNetB3',LEARNING_RATE=1e-05):   

    if MODEL == '8outputs': 

        # Entrada de imagen con 1 canal
        input_image = Input(shape=(300, 300, 1), name='input_image')

        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior

        if name_model.startswith('EfficientNet'):
            base_model = EfficientNetB3(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        elif  name_model.startswith('ConvNeXtBase'):
            base_model = ConvNeXtBase(input_shape=(300, 300, 3), include_top=False, weights='imagenet') 
        elif  name_model.startswith('ResNet50'):
            base_model = ResNet50(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        x_combined = Dense(128, activation='relu')(x)
        dropout_layer = Dropout(0.3)(x_combined)
        x_combined_h = Dense(64, activation='relu')(dropout_layer)
        x_combined_n = Dense(64, activation='relu')(dropout_layer)
        x_combined_t = Dense(64, activation='relu')(dropout_layer)
        x_combined_a = Dense(64, activation='relu')(dropout_layer)
        x_combined_p = Dense(64, activation='relu')(dropout_layer)
        x_combined_ll = Dense(64, activation='relu')(dropout_layer)
        x_combined_s = Dense(64, activation='relu')(dropout_layer)
        x_combined_ul = Dense(64, activation='relu')(dropout_layer)

        output_h = Dense(1, activation='sigmoid', name='output_h')(x_combined_h)
        output_n = Dense(1, activation='sigmoid', name='output_n')(x_combined_n)
        output_t = Dense(1, activation='sigmoid', name='output_t')(x_combined_t)
        output_a = Dense(1, activation='sigmoid', name='output_a')(x_combined_a)
        output_p = Dense(1, activation='sigmoid', name='output_p')(x_combined_p)
        output_ll = Dense(1, activation='sigmoid', name='output_ll')(x_combined_ll)
        output_s = Dense(1, activation='sigmoid', name='output_s')(x_combined_s)
        output_ul = Dense(1, activation='sigmoid', name='output_ul')(x_combined_ul)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=[input_image], outputs=[output_h,output_n,output_t,output_a,output_p,output_ll,output_s,output_ul])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={ 'output_h': 'binary_crossentropy',
                    'output_n': 'binary_crossentropy',
                    'output_t': 'binary_crossentropy',
                    'output_a': 'binary_crossentropy',
                    'output_p': 'binary_crossentropy',
                    'output_ll': 'binary_crossentropy',
                    'output_s': 'binary_crossentropy',
                    'output_ul': 'binary_crossentropy'

            },
            metrics={
                    'output_h': [BinaryAccuracy(), AUC()],
                    'output_n': [BinaryAccuracy(), AUC()],
                    'output_t': [BinaryAccuracy(), AUC()],
                    'output_a': [BinaryAccuracy(), AUC()],
                    'output_p': [BinaryAccuracy(), AUC()],
                    'output_ll': [BinaryAccuracy(), AUC()],
                    'output_s': [BinaryAccuracy(), AUC()],
                    'output_ul': [BinaryAccuracy(), AUC()]
                }
            )
    elif MODEL == '8outputs_SC': 

        # Entrada de imagen con 1 canal
        input_image = Input(shape=(300, 300, 1), name='input_image')

        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior

        if name_model.startswith('EfficientNet'):
            base_model = EfficientNetB3(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        elif  name_model.startswith('ConvNeXtBase'):
            base_model = ConvNeXtBase(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        elif  name_model.startswith('ResNet50'):
            base_model = ResNet50(input_shape=(300, 300, 3), include_top=False, weights='imagenet')

        
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        output_h = Dense(1, activation='sigmoid', name='output_h')(x)
        output_n = Dense(1, activation='sigmoid', name='output_n')(x)
        output_t = Dense(1, activation='sigmoid', name='output_t')(x)
        output_a = Dense(1, activation='sigmoid', name='output_a')(x)
        output_p = Dense(1, activation='sigmoid', name='output_p')(x)
        output_ll = Dense(1, activation='sigmoid', name='output_ll')(x)
        output_s = Dense(1, activation='sigmoid', name='output_s')(x)
        output_ul = Dense(1, activation='sigmoid', name='output_ul')(x)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=[input_image], outputs=[output_h,output_n,output_t,output_a,output_p,output_ll,output_s,output_ul])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={ 'output_h': 'binary_crossentropy',
                    'output_n': 'binary_crossentropy',
                    'output_t': 'binary_crossentropy',
                    'output_a': 'binary_crossentropy',
                    'output_p': 'binary_crossentropy',
                    'output_ll': 'binary_crossentropy',
                    'output_s': 'binary_crossentropy',
                    'output_ul': 'binary_crossentropy'

            },
            metrics={
                    'output_h': [BinaryAccuracy(), AUC()],
                    'output_n': [BinaryAccuracy(), AUC()],
                    'output_t': [BinaryAccuracy(), AUC()],
                    'output_a': [BinaryAccuracy(), AUC()],
                    'output_p': [BinaryAccuracy(), AUC()],
                    'output_ll': [BinaryAccuracy(), AUC()],
                    'output_s': [BinaryAccuracy(), AUC()],
                    'output_ul': [BinaryAccuracy(), AUC()]
                }
            )
    elif MODEL=='8outputs_DO':

        # Entrada de 1 canal (grayscale)
        input_image = Input(shape=(300, 300, 1), name='input_image')

        # Convertir de 1 a 3 canales para modelos pre-entrenados
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Seleccionar el modelo base
        name_model = 'EfficientNetB3'  # Cambiar según el modelo deseado

        if name_model.startswith('EfficientNet'):
            base_model = EfficientNetB3(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        elif name_model.startswith('ResNet50'):
            base_model = ResNet50(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
        else:
            raise ValueError(f'Modelo {name_model} no reconocido')

        # Hacer entrenables todas las capas del modelo base
        base_model.trainable = True

        # Aplicar modelo base
        x = base_model(x)

        # Capa Global Average Pooling y Dropout
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)  

        # Capas densas compartidas
        x_combined = Dense(128, activation='relu')(x)
        x_combined = Dropout(0.3)(x_combined)

        # Ramas de salida para cada predicción
        outputs = {}
        for region in ['h', 'n', 't', 'a', 'p', 'll', 's', 'ul']:
            branch = Dense(64, activation='relu')(x_combined)
            outputs[f'output_{region}'] = Dense(1, activation='sigmoid', name=f'output_{region}')(branch)

        # Definir modelo con múltiples salidas
        model = Model(inputs=input_image, outputs=list(outputs.values()))
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={ 'output_h': 'binary_crossentropy',
                    'output_n': 'binary_crossentropy',
                    'output_t': 'binary_crossentropy',
                    'output_a': 'binary_crossentropy',
                    'output_p': 'binary_crossentropy',
                    'output_ll': 'binary_crossentropy',
                    'output_s': 'binary_crossentropy',
                    'output_ul': 'binary_crossentropy'

            },
            metrics={
                    'output_h': [BinaryAccuracy(), AUC()],
                    'output_n': [BinaryAccuracy(), AUC()],
                    'output_t': [BinaryAccuracy(), AUC()],
                    'output_a': [BinaryAccuracy(), AUC()],
                    'output_p': [BinaryAccuracy(), AUC()],
                    'output_ll': [BinaryAccuracy(), AUC()],
                    'output_s': [BinaryAccuracy(), AUC()],
                    'output_ul': [BinaryAccuracy(), AUC()]
                }
            )



    else:
        print('Invalid model name')


    return model

