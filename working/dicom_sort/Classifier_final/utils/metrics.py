import tensorflow as tf
import tensorflow.keras.backend as K


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=1)  # Predicción a clases
        self.precision.update_state(y_true, y_pred_classes, sample_weight)
        self.recall.update_state(y_true, y_pred_classes, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


class PrecisionCustom(tf.keras.metrics.Metric):
    def __init__(self, name='precision_custom', **kwargs):
        super(PrecisionCustom, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=1)  # Convertir predicciones a clases
        y_true = tf.squeeze(y_true)  # Asegurar forma compatible
        
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred_classes, y_true), tf.equal(y_true, 1)), dtype=tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred_classes, y_true), tf.equal(y_true, 0)), dtype=tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        return self.tp / (self.tp + self.fp + K.epsilon())

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)


class RecallCustom(tf.keras.metrics.Metric):
    def __init__(self, name='recall_custom', **kwargs):
        super(RecallCustom, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true = tf.squeeze(y_true)
        
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred_classes, y_true), tf.equal(y_true, 1)), dtype=tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred_classes, y_true), tf.equal(y_true, 1)), dtype=tf.float32))

        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + K.epsilon())

    def reset_states(self):
        self.tp.assign(0)
        self.fn.assign(0)


class AUCCustom(tf.keras.metrics.Metric):
    def __init__(self, name='auc_custom', **kwargs):
        super(AUCCustom, self).__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        self.auc.update_state(y_true, y_pred[:, 1])  # Se usa la probabilidad de la clase positiva

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()