import cv2
import numpy as np
import tensorflow as tf

TOWER_NAME = 'tower'
IMG_SIZE = 256
CROP_IMG_SIZE = 227


class InceptionAgeAnalyzer:
    def __init__(self, model_config):
        """
        Initialize inception v3 model for age analysis on face
        :param model_config:
               model_config (dict) - 'model_path' is the path to the trained model
                                  'lbl_encoding' is the encoding of the labels
                                                 in the trained model.
        """
        export_path = model_config['model_path']
        predict_tensor_name = model_config['predict_tensor']
        input_tensor_name = model_config['input_tensor']
        self._lbl_encoding = dict((v, k) for k, v in model_config['lbl_encoding'].items())
        with tf.Graph().as_default() as graph:
            sess = tf.Session()
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
            predict_tensor = graph.get_tensor_by_name(predict_tensor_name)

            self.prediction_fn = lambda resized_face: sess.run(predict_tensor,
                                                               feed_dict={input_tensor_name: resized_face})

    def analyze_age(self, face_data):

        """
        Estimate the age by analyzing face data
        :param face_data: np array of face data
        :return: a tuple of age estimation, probability
        """
        resized_face = cv2.resize(face_data, (IMG_SIZE, IMG_SIZE))

        pred_results = self.prediction_fn(resized_face)
        # predict_tensor = self.graph.get_tensor_by_name(self.predict_tensor_name)
        # pred_results = self.session.run(predict_tensor, feed_dict={self.input_tensor_name: resized_face})
        output = pred_results[0]
        batch_sz = pred_results.shape[0]
        for i in range(1, batch_sz):
            output = output + pred_results[i]
        output /= batch_sz
        best = np.argmax(output)
        best_prob = output[best]
        best_group = self._lbl_encoding[best]
        output[best] = 0
        second_best = np.argmax(output)
        second_best_prob = output[second_best]
        second_best_group = self._lbl_encoding[second_best]

        sum_prob = best_prob + second_best_prob
        # get the weighted age estimation based on probability of best and second best age group
        if second_best_group[0] > best_group[0]:
            after_group = self._lbl_encoding[best + 1]
            start = 0.5 * best_group[1] + 0.5 * best_group[0]
            end = 0.5 * after_group[0] + 0.5 * after_group[1]
            weighted_age = int((start * best_prob + end * second_best_prob) / sum_prob)
        else:
            before_group = self._lbl_encoding[best - 1]
            if best == len(self._lbl_encoding) - 1:
                start = before_group[1]
                end = best_group[1]
                weighted_age = int(start + best_prob * (end - start))
            else:
                start = 0.5 * before_group[1] + 0.5 * before_group[0]
                end = 0.5 * best_group[1] + 0.5 * best_group[0]
                weighted_age = int((end * best_prob + start * second_best_prob) / sum_prob)

        return weighted_age, best_prob
