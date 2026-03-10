from tensorflow.keras.applications import ResNet50,EfficientNetB3,ConvNeXtBase,VGG19,VGG16,EfficientNetV2B3,EfficientNetV2M
from tensorflow.keras.layers import Concatenate, Dense, Input, GlobalAveragePooling2D,Conv2D,Multiply,Dropout
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils.metrics import *
from tensorflow.keras.metrics import AUC, Precision, Recall, AUC, Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Input, Conv2D,  GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def crear_modelo(MODEL,TRAINING_MODE,ETIQUETAS_DICOM,CLASSES_W=0,LEARNING_RATE=1e-5, name='test',function_act=[],loss=[],name_model='EfficientNetB0'):   

    if MODEL == 'Reconected': 

        # Entrada de imagen con 1 canal
        input_image = Input(shape=(224, 224, 1), name='input_image')

        # Adaptar MobileNet para aceptar imágenes de 1 canal
        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        base_model = EfficientNetV2B3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        x = base_model(x)

        for layer in base_model.layers:
            if TRAINING_MODE=='scratch':       
                if layer.name.startswith('block'):
                    layer.trainable =True 
            elif TRAINING_MODE=='TL5':       
                if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                    layer.trainable =True 

        x = GlobalAveragePooling2D()(x)

        input_values = Input(shape=(len(ETIQUETAS_DICOM),), name='input_values')  # Todas las características numéricas
        input_masks = Input(shape=(len(ETIQUETAS_DICOM),), name='input_masks')  # Todas las máscaras numéricas

        # Función personalizada para aplicar la lógica de la máscara
        def apply_mask(inputs):
            input_values, input_masks = inputs
            return tf.where(input_masks == 1, input_values, tf.constant(-1.0, dtype=tf.float32))

        # Aplicar la máscara usando Lambda
        masked_inputs = Lambda(apply_mask, name='masked_inputs')([input_values, input_masks])


        # Aplicar la máscara
        #masked_inputs = Multiply(name='masked_inputs')([input_values, input_masks])

        # Procesar los datos numéricos
        x_num = Dense(64, activation='relu')(masked_inputs)
        x_num = Dropout(0.4)(x_num)
        x_num = Dense(32, activation='relu')(x_num)
        x_num = Dropout(0.4)(x_num)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        x_combined = Concatenate(name='fusion_layer')([x, x_num])
        dropout_layer = Dropout(0.4)(x_combined)
        x_combined = Dense(128, activation='relu')(dropout_layer)
        x_combined_W = Dense(32, activation='relu')(x_combined)
        x_combined_fs = Dense(64, activation='relu')(x_combined)
        x_combined_c = Dense(64, activation='relu')(x_combined)

        output_w = Dense(len(CLASSES_W), activation='softmax', name='output_w')(x_combined_W)

        combined_fs = Concatenate()([output_w, x_combined_fs])
        x2 = Dense(32, activation='relu')(combined_fs)
        output_fs = Dense(1, activation='sigmoid', name='output_fs')(x2)

        combined_c = Concatenate()([output_fs,output_w, x_combined_c])
        x3 = Dense(32, activation='relu')(combined_c)
        output_c = Dense(1, activation='sigmoid', name='output_c')(x3)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=[input_image, input_values, input_masks], outputs=[output_w,output_fs,output_c])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={
                'output_w': 'sparse_categorical_crossentropy',
                'output_fs': 'binary_crossentropy',
                'output_c': 'binary_crossentropy'
            },
            metrics={
                'output_w': 'accuracy',
                'output_fs': 'accuracy',
                'output_c': 'accuracy'
            }
        )

    elif MODEL == '3outputs': 
  
        # Entrada de imagen con 1 canal
        input_image = Input(shape=(224, 224, 1), name='input_image')

        # Adaptar MobileNet para aceptar imágenes de 1 canal
        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        base_model = EfficientNetV2B3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

        for layer in base_model.layers:
            if TRAINING_MODE=='scratch':       
                if layer.name.startswith('block'):
                    layer.trainable =True 
            elif TRAINING_MODE=='TL5':       
                if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                    layer.trainable =True 

        x = base_model(x)
        x = GlobalAveragePooling2D()(x)

        input_values = Input(shape=(len(ETIQUETAS_DICOM),), name='input_values')  # Todas las características numéricas
        input_masks = Input(shape=(len(ETIQUETAS_DICOM),), name='input_masks')  # Todas las máscaras numéricas

        # Aplicar la máscara
        masked_inputs = Multiply(name='masked_inputs')([input_values, input_masks])

        # Procesar los datos numéricos
        x_num = Dense(64, activation='relu')(masked_inputs)
        x_num = Dropout(0.4)(x_num)
        x_num = Dense(32, activation='relu')(x_num)
        x_num = Dropout(0.3)(x_num)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        x_combined = Concatenate(name='fusion_layer')([x, x_num])
        dropout_layer = Dropout(0.4)(x_combined)
        x_combined = Dense(128, activation='relu')(dropout_layer)
        x_combined_W = Dense(64, activation='relu')(x_combined)
        x_combined_fs = Dense(64, activation='relu')(x_combined)
        x_combined_c = Dense(64, activation='relu')(x_combined)
        output_w = Dense(len(CLASSES_W), activation='softmax', name='output_w')(x_combined_W)
        output_fs = Dense(1, activation='sigmoid', name='output_fs')(x_combined_fs)
        output_c = Dense(1, activation='sigmoid', name='output_c')(x_combined_c)

        model = Model(inputs=[input_image, input_values, input_masks], outputs=[output_w,output_fs,output_c])

        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={'output_w': 'sparse_categorical_crossentropy',
                'output_fs': 'binary_crossentropy',
                'output_c': 'binary_crossentropy'

            },
            metrics={
                'output_w': 'accuracy',
                'output_fs': 'accuracy',
                'output_c': 'accuracy'
            })

    elif MODEL == '2outputs': 
  
        # Entrada de imagen con 1 canal
        input_image = Input(shape=(224, 224, 1), name='input_image')

        # Adaptar MobileNet para aceptar imágenes de 1 canal
        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        base_model = EfficientNetB3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

        for layer in base_model.layers:
            if TRAINING_MODE=='scratch':       
                if layer.name.startswith('block'):
                    layer.trainable =True 
            elif TRAINING_MODE=='TL5':       
                if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                    layer.trainable =True 

        x = base_model(x)
        x = GlobalAveragePooling2D()(x)

        input_values = Input(shape=(len(ETIQUETAS_DICOM),), name='input_values')  # Todas las características numéricas
        input_masks = Input(shape=(len(ETIQUETAS_DICOM),), name='input_masks')  # Todas las máscaras numéricas

        # Aplicar la máscara
        masked_inputs = Multiply(name='masked_inputs')([input_values, input_masks])

        # Procesar los datos numéricos
        x_num = Dense(64, activation='relu')(masked_inputs)
        x_num = Dropout(0.4)(x_num)
        x_num = Dense(32, activation='relu')(x_num)
        x_num = Dropout(0.3)(x_num)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        x_combined = Concatenate(name='fusion_layer')([x, x_num])
        dropout_layer = Dropout(0.4)(x_combined)
        x_combined = Dense(128, activation='relu')(dropout_layer)
        x_combined_fs = Dense(64, activation='relu')(x_combined)
        x_combined_c = Dense(64, activation='relu')(x_combined)
        output_fs = Dense(1, activation='sigmoid', name='output_fs')(x_combined_fs)
        output_c = Dense(1, activation='sigmoid', name='output_c')(x_combined_c)

        model = Model(inputs=[input_image, input_values, input_masks], outputs=[output_fs,output_c])

        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={
                'output_fs': 'binary_crossentropy',
                'output_c': 'binary_crossentropy'

            },
            metrics={
                'output_fs': 'accuracy',
                'output_c': 'accuracy'
            })
        
    
    elif MODEL == '1output':

                # Entrada de imagen con 1 canal
        input_image = Input(shape=(224, 224, 1), name='input_image')

        # Adaptar MobileNet para aceptar imágenes de 1 canal
        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        base_model = EfficientNetB3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

        
        for layer in base_model.layers:
            if TRAINING_MODE=='scratch':       
                if layer.name.startswith('block'):
                    layer.trainable =True 
            elif TRAINING_MODE=='TL5':       
                if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                    layer.trainable =True 


        x = base_model(x)
        x = GlobalAveragePooling2D()(x)

        input_values = Input(shape=(len(ETIQUETAS_DICOM),), name='input_values')  # Todas las características numéricas
        input_masks = Input(shape=(len(ETIQUETAS_DICOM),), name='input_masks')  # Todas las máscaras numéricas

        # Función personalizada para aplicar la lógica de la máscara
        def apply_mask(inputs):
            input_values, input_masks = inputs
            return tf.where(input_masks == 1, input_values, tf.constant(-1.0, dtype=tf.float32))

        # Aplicar la máscara usando Lambda
        masked_inputs = Lambda(apply_mask, name='masked_inputs')([input_values, input_masks])


        # Aplicar la máscara
        #masked_inputs = Multiply(name='masked_inputs')([input_values, input_masks])

        # Procesar los datos numéricos
        x_num = Dense(64, activation='relu')(masked_inputs)
        x_num = Dropout(0.4)(x_num)
        x_num = Dense(32, activation='relu')(x_num)
        x_num = Dropout(0.3)(x_num)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        x_combined = Concatenate(name='fusion_layer')([x, x_num])
        dropout_layer = Dropout(0.4)(x_combined)
        x_combined = Dense(128, activation='relu')(dropout_layer)
        x_combined_W = Dense(64, activation='relu')(x_combined)
        output_w = Dense(len(CLASSES_W), activation=function_act, name= f'output_{name}')(x_combined_W)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=[input_image, input_values, input_masks], outputs=[output_w])
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={f'output_{name}': loss
            },
            metrics={
                f'output_{name}': 'accuracy'})
        
       
    elif MODEL == '1output_only':

        # Entrada de imagen con 1 canal
        input_image = Input(shape=(224, 224, 1), name='input_image')

        # Adaptar MobileNet para aceptar imágenes de 1 canal
        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        if name_model.startswith('EfficientNetV2'):
            base_model = EfficientNetV2M(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if layer.name.startswith('block'):
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                        layer.trainable =True 
        elif name_model.startswith('EfficientNet'):
            base_model = EfficientNetB3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if layer.name.startswith('block'):
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                        layer.trainable =True 
        elif  name_model.startswith('ConvNeXtBase'):
            base_model = ConvNeXtBase(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if 'stage' in layer.name:
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'stage_3' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('ResNet50'):
            base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if 'block' in layer.name:
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block3' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('VGG19'):
            base_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for index,layer in enumerate(base_model.layers):
                if TRAINING_MODE=='scratch':       
                    layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block4' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('VGG16'):
            base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block4' in layer.name:
                        layer.trainable =True 


        x = base_model(x)
        x = GlobalAveragePooling2D()(x)

        input_values = Input(shape=(len(ETIQUETAS_DICOM),), name='input_values')  # Todas las características numéricas
        input_masks = Input(shape=(len(ETIQUETAS_DICOM),), name='input_masks')  # Todas las máscaras numéricas

        # Función personalizada para aplicar la lógica de la máscara
        def apply_mask(inputs):
            input_values, input_masks = inputs
            return tf.where(input_masks == 1, input_values, tf.constant(-1.0, dtype=tf.float32))

        # Aplicar la máscara usando Lambda
        masked_inputs = Lambda(apply_mask, name='masked_inputs')([input_values, input_masks])


        # Aplicar la máscara
        #masked_inputs = Multiply(name='masked_inputs')([input_values, input_masks])

        # Procesar los datos numéricos
        x_num = Dense(64, activation='relu')(masked_inputs)
        x_num = Dropout(0.3)(x_num)
        x_num = Dense(32, activation='relu')(x_num)
        x_num = Dropout(0.3)(x_num)

        # ============================
        # FUSIÓN Y SALIDA
        # ============================
        x_combined = Concatenate(name='fusion_layer')([x, x_num])
        dropout_layer = Dropout(0.3)(x_combined)
        x_combined = Dense(128, activation='relu')(dropout_layer)
        x_combined_W = Dense(64, activation='relu')(x_combined)
        output_w = Dense(len(CLASSES_W), activation=function_act, name= f'output_{name}')(x_combined_W)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=[input_image, input_values, input_masks], outputs=[output_w])
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={f'output_{name}': loss
            },
            metrics=[
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall'),]
    )
        """metrics={
                    f'output_{name}': [
                        'accuracy',  # Precisión estándar
                        Precision(name="precision"),
                        Recall(name="recall"),
                        AUC(name="auc"),
                        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")  ]   })
            """
  # Exactitud binaria
                   

                     #, F1Score(),PrecisionCustom(),RecallCustom(),AUCCustom()]

    elif MODEL=='1outputs_DO':

        # Entrada de 1 canal (grayscale)
        input_image = Input(shape=(300, 300, 1), name='input_image')

        # Convertir de 1 a 3 canales para modelos pre-entrenados
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Seleccionar el modelo base
        name_model = 'EfficientNetB3'  # Cambiar según el modelo deseado

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        if name_model.startswith('EfficientNet'):
            base_model = EfficientNetB3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if layer.name.startswith('block'):
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                        layer.trainable =True 
        elif  name_model.startswith('ConvNeXtBase'):
            base_model = ConvNeXtBase(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if 'stage' in layer.name:
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'stage_3' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('ResNet50'):
            base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if 'block' in layer.name:
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block3' in layer.name:
                        layer.trainable =True 


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

    elif MODEL=='1outputs_only_DO':

        # Entradas
        dropout_rate=0.2
        input_image = layers.Input(shape=(300, 300, 1), name='input_image')
        x = layers.Conv2D(3, (1, 1), padding='same', activation='relu')(input_image)  

        # 🔹 **Bloque Inicial**
        #x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(input_image)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # 🔹 **Bloque 1**
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(16, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)  # Agregar Dropout

        # 🔹 **Bloque 2**
        residual=layers.Conv2D(24, (1, 1), padding="same", use_bias=False)(x)
        x = layers.Conv2D(64, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(24, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])  # Suma residual
        x = layers.Dropout(dropout_rate)(x)

        # 🔹 **Bloque 3**
        residual = layers.Conv2D(40, (1, 1), padding="same", use_bias=False)(x)
        x = layers.Conv2D(72, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(40, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(dropout_rate)(x)

        # 🔹 **Bloque 4**
        residual = layers.Conv2D(80, (1, 1), padding="same", use_bias=False)(x)
        x = layers.Conv2D(120, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(80, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(dropout_rate)(x)

        # 🔹 **Bloque 5**
        residual = layers.Conv2D(112, (1, 1), padding="same", use_bias=False)(x)
        x = layers.Conv2D(200, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(112, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(dropout_rate)(x)

        # 🔹 **Bloque 6**
        residual = layers.Conv2D(192, (1, 1), padding="same", use_bias=False)(x)
        x = layers.Conv2D(304, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(192, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(dropout_rate)(x)

        # 🔹 **Bloque 7**
        residual = layers.Conv2D(320, (1, 1), padding="same", use_bias=False)(x)
        x = layers.Conv2D(512, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(320, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(dropout_rate)(x)

        # 🔹 **Capa Final**
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1280, activation="relu")(x)
        x = layers.Dropout(0.4)(x)  # Dropout final


        # Entrada numérica
        input_values = layers.Input(shape=(len(ETIQUETAS_DICOM),), name='input_values')  # Datos numéricos
        input_masks = layers.Input(shape=(len(ETIQUETAS_DICOM),), name='input_masks')  # Máscaras numéricas

        def apply_mask(inputs):
            input_values, input_masks = inputs
            return tf.where(input_masks == 1, input_values, tf.constant(-1.0, dtype=tf.float32))

        # Aplicar máscara a los valores numéricos
        masked_inputs = layers.Lambda(apply_mask, name='masked_inputs')([input_values, input_masks])

        # Procesar las características numéricas
        x_num = layers.Dense(64, activation='relu')(masked_inputs)
        x_num = layers.Dropout(0.3)(x_num)
        x_num = layers.Dense(32, activation='relu')(x_num)
        x_num = layers.Dropout(0.3)(x_num)

        # Fusionar las salidas de la imagen y las características numéricas
        x_combined = layers.Concatenate(name='fusion_layer')([x, x_num])
        dropout_layer = layers.Dropout(0.3)(x_combined)
        x_combined = layers.Dense(128, activation='relu')(dropout_layer)
        x_combined_W = layers.Dense(64, activation='relu')(x_combined)

        # Salida final
        output_w = Dense(len(CLASSES_W), activation=function_act, name= f'output_{name}')(x_combined_W)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=[input_image, input_values, input_masks], outputs=[output_w])
        model.summary()
        # Compilación del modelo
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={f'output_{name}': loss
            },
            metrics={f'output_{name}': 'accuracy'})
 
  # Exactitud binaria
                   

                     #, F1Score(),PrecisionCustom(),RecallCustom(),AUCCustom()]


    else:
        print('Invalid model name')


    return model

def crear_modelo_img(MODEL,TRAINING_MODE,CLASS,LEARNING_RATE=1e-5, name='test',image_size=[224,224],function_act=[],loss=[],name_model='EfficientNetB0'):   

    if MODEL == '1output_img':

        # Entrada de imagen con 1 canal
        input_image = Input(shape=(image_size[0], image_size[1], 1), name='input_image')

        # Adaptar MobileNet para aceptar imágenes de 1 canal
        # Agregar una capa Conv2D para convertir de 1 a 3 canales
        x = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name='reescale')(input_image)

        # Cargar MobileNet preentrenado en ImageNet, sin la capa superior
        if  name_model.startswith('EfficientNetV2'):
            base_model = EfficientNetV2M(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block4' in layer.name:
                        layer.trainable =True 
        elif name_model.startswith('EfficientNet'):
            base_model = EfficientNetB3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if layer.name.startswith('block'):
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if layer.name.startswith('block5') or layer.name.startswith('block6') or layer.name.startswith('block7'):
                        layer.trainable =True 
        elif  name_model.startswith('ConvNeXtBase'):
            base_model = ConvNeXtBase(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if 'stage' in layer.name:
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'stage_3' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('ResNet50'):
            base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    if 'block' in layer.name:
                        layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block3' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('VGG19'):
            base_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for index,layer in enumerate(base_model.layers):
                if TRAINING_MODE=='scratch':       
                    layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block4' in layer.name:
                        layer.trainable =True 
        elif  name_model.startswith('VGG16'):
            base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if TRAINING_MODE=='scratch':       
                    layer.trainable =True 
                elif TRAINING_MODE=='TL5':       
                    if 'block4' in layer.name:
                        layer.trainable =True 



        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_w = Dense(len(CLASS), activation=function_act, name= f'output_{name}')(x)

        # ============================
        # DEFINIR EL MODELO
        # ============================
        model = Model(inputs=input_image, outputs=output_w)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss={f'output_{name}': loss
            },
            metrics=[
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall'),
    ])
        """metrics={
                    f'output_{name}': [
                        'accuracy',  # Precisión estándar
                        Precision(name="precision"),
                        Recall(name="recall"),
                        AUC(name="auc"),
                        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")  ]   })
            """


    else:
        print('Invalid model name')


    return model


def add_dropout_after_blocks(input_image,base_model, dropout_rate=0.5):
    """
    Modifica el modelo base EfficientNetB3 añadiendo Dropout después de cada bloque convolucional.
    """
    x = input_image
    for layer in base_model.layers:
        x = layer(x)
        if "add" in layer.name:  # Normalmente las conexiones residuales de EfficientNet terminan en "add"
            x = layers.Dropout(dropout_rate)(x)

    
    return x