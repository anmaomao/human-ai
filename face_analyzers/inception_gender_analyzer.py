import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

TOWER_NAME = 'tower'
IMG_SIZE = 256
CROP_IMG_SIZE = 227
BATCH_SIZE = 12


class InceptionGenderAnalyzer:
    def __init__(self, model_config):
        """
        Initialize inception v3 model for gender analysis on face
        :param model_config:
               model_config (dict) - 'model_path' is the path to the trained model
                                  'lbl_encoding' is the encoding of the labels
                                                 in the trained model.
        """
        export_path = model_config['model_path']
        predict_tensor_name = 'prediction:0'
        input_tensor_name = 'image:0'
        self._lbl_encoding = model_config['lbl_encoding']
        with tf.Graph().as_default() as graph:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
            predict_tensor = graph.get_tensor_by_name(predict_tensor_name)

            self.prediction_fn = lambda resized_face: sess.run(predict_tensor,
                                                               feed_dict={input_tensor_name: resized_face})

    def analyze_gender(self, face_data):
        """
        Estimate the gender by analyzing face data
        :param face_data_list: np array of a list of face data
        :return: a tuple of gender estimation, probability
        """
        resized_face = cv2.resize(face_data, (IMG_SIZE, IMG_SIZE))
        # pred_results = self.prediction_fn({"image": resized_face})['prediction']
        pred_results = self.prediction_fn(resized_face)
        output = pred_results[0]
        for j in range(1, BATCH_SIZE):
            output = output + pred_results[j]
        output /= BATCH_SIZE
        decoded_gender_pred = dict([(name, output[idx]) for name, idx in
                     self._lbl_encoding.items()])

        return decoded_gender_pred
